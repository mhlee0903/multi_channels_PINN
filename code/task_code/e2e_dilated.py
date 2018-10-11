import sys
import pandas as pd
import numpy as np
import time
import random
import gc
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, EarlyStopping

from scipy import interp
from keras.layers import Dense, concatenate,Input,Dropout,AlphaDropout
from keras.optimizers import SGD, Adadelta, Nadam, Adagrad, Adam, RMSprop
from keras.utils import np_utils
from keras.models import Model
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, matthews_corrcoef, precision_recall_curve, average_precision_score
import math
import models_cnn_dilated as m_
########
from keras import backend as K
from keras.models import load_model

import os
import datetime
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='raw data model based on CNN')
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--GPU', type=str, default='0')

    parser.add_argument('--TRAIN_batch', type=int, default=1024)
    parser.add_argument('--TRAIN_epoch', type=int, default=201)
    parser.add_argument('--TRAIN_lr', type=float, default=0.0005)

    parser.add_argument('--TRANS_opt', type=str, default='')
    
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--nb_filters', type=int, default=64)
    parser.add_argument('--nb_kernels', type=int, default=4)
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--RE_model', type=str, default='')
    parser.add_argument('--RE_epoch', type=int, default=0)

    parser.add_argument('--TAXONOMY', type=str, default='human')
    parser.add_argument('--MODEL_str', type=str, default='3-1')
    args = parser.parse_args()

GPU_device = '/gpu:'+args.GPU

def set_callback(TRANS_opt):
    callback_list = []
    callback_list.append(task_val_call(model=model, path_w=ckpt_path, n_addL=0, path_csv=dir_ret, batch_test = 1024))
    
    return callback_list

def history_csv(history, dir_csv):
    key_list = history.history.keys()
    ret_list=[]
    for key in key_list:
        ret_list.append( history.history[key])

    pd.DataFrame(np.array(ret_list).T, columns=key_list).to_csv(dir_csv)



def data_valid( pair_dataset, table_aa, table_smile ):
    # get protein 
    temp_prot=pair_dataset['protein'].values  
    X_prot = table_aa.loc[temp_prot].values

    # get compound
    temp_com = pair_dataset['mol_id'].values  
    X_mol = table_smile.loc[temp_com].values

    X_in=[X_prot,  X_mol]
    y_oneHot = np_utils.to_categorical(pair_dataset['label'].values)
    return [X_in, y_oneHot]

def eval_N_csv(y_oneHot, DB, csv_list, model, epoch_transfer, epoch_train, batch_size, X_in, path_csv, taskName=None):
    y_oneHot = y_oneHot
    y_test_raw = categorical_probas_to_classes(y_oneHot)

    y_probas = model.predict(X_in, batch_size=batch_size, verbose=0)

    write_result( y_test_raw, y_probas, 
                 cvscores= csv_list, taskName=taskName, path = path_csv,
                 epoch_train=epoch_train, epoch_transfer=epoch_transfer, DB=DB)

    del y_probas
    del y_test_raw
    del X_in    


def _mcc(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def write_result(y_raw, y_probas, cvscores ,path ,DB, taskName='All', epoch_train=None, epoch_transfer=None):
    y_pred = categorical_probas_to_classes(y_probas)
    y_target=y_raw[:len(y_pred)]

    roc_auc = roc_auc_score(y_target, y_probas[:,1])
    pre, rec, thresholds = precision_recall_curve(y_target, y_probas[:,1])
    prc = auc(rec, pre)

    acc, precision, npv, sensitivity, specificity, mcc, f1, tp, fp, tn, fn= calculate_performace(len(y_pred), y_target, y_pred)
    cvscores.append([acc, precision,npv, sensitivity, specificity, mcc,roc_auc, prc, 
                    tp, fp, tn, fn, 
                    DB,taskName, epoch_train, epoch_transfer] )
    
    cv_df = pd.DataFrame(cvscores, columns=['acc', 'precision','npv', 'sensitivity', 'specificity', 'mcc','roc_auc', 'prc',
                                            'true pos','false pos','true neg','false neg', 
                                            'DB','taskName','epoch_train', 'epoch_transfer' ])

    cv_df.to_csv(path+'{DB}.csv'.format(DB=DB))


def train_generator(batch_size, pair_dataset, table_aa, table_smile):
    data_len=len(pair_dataset)
    total_idx = np.arange(data_len)
    np.random.shuffle(total_idx)
    y_raw = pair_dataset['label'].values
    y_oneHot = np_utils.to_categorical(y_raw)

    while True:
        for i in range(int( data_len//batch_size)):
            idx_start = i*batch_size
            idx_end   = (i+1)*batch_size
            idx_batch = total_idx[idx_start:idx_end]

            # get protein 
            temp_prot=pair_dataset['protein'].values[idx_batch]  
            X_prot = table_aa.loc[temp_prot].values
            
            # get compound
            temp_com = pair_dataset['mol_id'].values[idx_batch]  
            X_mol = table_smile.loc[temp_com].values
            
            X_in=[X_prot,  X_mol]

            yield X_in, y_oneHot[idx_batch]


def calculate_performace(test_num,  y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)

    mcc=matthews_corrcoef( y_true , y_pred )
    f1=2 * (precision * sensitivity) / (precision + sensitivity)
    return acc, precision,npv, sensitivity, specificity, mcc, f1, tp, fp, tn, fn


def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def scaled_df(scaler, df):
    return pd.DataFrame(scaler.transform(df), index=df.index.values)

def set_path( base_path, epoch_tr=None, addL=None, task=None):
    path = base_path 
    if addL is not None:
        path = base_path+"/Trans_{task}_on_train_epoch{epoch_tr}_addL{addL}/".format(task=task, epoch_tr=epoch_tr ,addL=addL)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# ## Change Keras Backend as TF
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
        

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

back_end='tensorflow'
set_keras_backend(back_end)
K.set_session(sess)
print( 'Backend is changed: '+ K.backend())

###################################################

verbose = 1

MODEL_dic = {'3-1':m_.pcm3_1,'4-1':m_.pcm4_1, '5-1':m_.pcm5_1,'6-1':m_.pcm6_1}


# 2.1 Get task pair data
task_fname_val ='../../data/task_data/task_pair_test.csv'
task_pair_val = pd.read_csv(task_fname_val, index_col=0, header=0)
task_fname_tr ='../../data/task_data/task_pair_train.csv'
task_pair_train = pd.read_csv(task_fname_tr, index_col=0, header=0)

table_aa    = pd.read_csv('../../data/transfer_data/transfer_table_prot_seq[700].csv', index_col=0)
table_smile = pd.read_csv('../../data/transfer_data/transfer_table_smile[100].csv', index_col=0)

task_val_X_in, task_val_y_oneHot = data_valid(
                                       pair_dataset=task_pair_val,
                                       table_aa=    table_aa,
                                       table_smile= table_smile )

class task_val_call(Callback):
    def __init__(self, model, path_csv,  n_addL, epoch_train=None, path_w= None, full_learning='', batch_test= 1024):        
        # set the CSV list
        self.CSV_task_te =[]
        
        # We set the model (non multi gpu) under an other name
        self.model_tr = model
        
        # set path for csv file
        self.path_csv = path_csv+'/val_direct/'
        if not os.path.exists(self.path_csv):
            os.makedirs(self.path_csv)     
        self.path_csv = self.path_csv+'/epoch{epoch}_'
        self.epoch_train = epoch_train

        # set path for weights

        self.path = path_w
        if self.path is not None :
            if not os.path.exists(self.path):
                os.makedirs(self.path)            
            self.path_save= self.path+"model_tr_epoch{epoch:d}.h5"
        self.batch = batch_test

        self.full_learning=full_learning
        self.n_addL=n_addL

    # def on_epoch_begin(self, epoch, logs=None, verbose=False):
        # if verbose is True:        
        #     str_model_info = para_info+'\ttask_val_on_epoch_begin@{}\tbased on trainE[{}/{:3d}]\t'.format(self.full_learning+'task',self.epoch_train, args.TRAIN_epoch)
        #     str_trainable_w= 'trainable weights:[{}]\t'.format(len(self.model_tr.trainable_weights))+'+'*len(self.model_tr.trainable_weights)
        #     print str_model_info, str_trainable_w

    def on_epoch_end(self, epoch, logs=None):
        # 1.4. Evaluate task_te
        eval_N_csv(y_oneHot= task_val_y_oneHot, X_in= task_val_X_in, model=self.model, path_csv=self.path_csv.format(epoch=epoch),
                   epoch_train=self.epoch_train, epoch_transfer=epoch, batch_size=self.batch, 
                   csv_list=self.CSV_task_te, DB='Task', taskName='task_te' )
        
        # 2. Save Weights
        if self.path is not None :
            self.model_tr.save(self.path_save.format(epoch=epoch), overwrite=True)

############################
steps_task = int(len(task_pair_train)//(args.TRAIN_batch))+1
nb_patience = 100
patience_task=int(nb_patience/2)
#############################

########### Model Info setting ##############
fname_temp=__file__

fName=fname_temp[:-3]

time_now = datetime.datetime.now().strftime('%m-%d %H:%M')

dir_model='./saved_models/'+fName+'/'# +para_info+'/'
dir_model=set_path(dir_model)

dir_ret = './results/'+fName+'/'# +para_info+'/'
dir_ret=set_path(dir_ret)
path_train = dir_ret+'train_model_history.csv'

########### Model Build! ##############
with tf.device('/cpu:0'):
    model= MODEL_dic[args.MODEL_str](dropout_value=0.5, 
                                     num_filters = args.nb_filters,
                                     embed_dim=args.embed_dim,
                                     kernel_size=args.nb_kernels)

if args.RE_model is not '':
    re_path = dir_model+args.RE_model
    model = load_model(re_path)
    print('re-load the model!\t'+args.RE_model)

with tf.device(GPU_device):
    parallel_model = multi_gpu_model(model, gpus=args.n_gpu)
    sgd = Adam( lr= args.TRAIN_lr , decay=1e-6)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', _mcc])

    # Setting the callback functions
    ckpt_path = dir_model
    callback_list = set_callback(args.TRANS_opt)
        
    history = parallel_model.fit_generator(
                                 generator=train_generator(
                                                batch_size= args.TRAIN_batch,
                                                pair_dataset=  task_pair_train,
                                                table_aa = table_aa,
                                                table_smile = table_smile )  ,
                                  steps_per_epoch= steps_task,
                                  epochs=args.TRAIN_epoch,
                                  verbose= verbose,
                                  callbacks= callback_list,
                                  max_queue_size=12,
                                  workers=2,
                                  use_multiprocessing=False, 
                                  shuffle=True,
                                  initial_epoch= args.RE_epoch,
                                  )
    history_csv(history, path_train)
