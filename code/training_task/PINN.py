import sys
import pandas as pd
import numpy as np
import time
import random
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, EarlyStopping

from keras.layers import Dense, concatenate,Input,Dropout,AlphaDropout
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, matthews_corrcoef, precision_recall_curve, average_precision_score
import math
import models as m_
########
from keras import backend as K
from keras.models import load_model

import os
import datetime
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCPINN')
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--GPU', type=str, default='0')
    parser.add_argument('--gpu_frac', type=float, default='1.0')

    parser.add_argument('--TRAIN_batch', type=int, default=1024)
    parser.add_argument('--TRAIN_epoch', type=int, default=500)
    parser.add_argument('--TRAIN_lr', type=float, default=0.0005)

    parser.add_argument('--PROT_type', type=str, default='H')
    parser.add_argument('--CMPD_type', type=str, default='H')
    parser.add_argument('--SL_length', type=int, default=2)
    parser.add_argument('--CL_length', type=int, default=1)

    parser.add_argument('--ecfp', type=int, default=0)
    parser.add_argument('--vec', type=int, default=1)
    parser.add_argument('--n_nodes', type=int, default=1024)

    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--nb_filters', type=int, default=16)

    parser.add_argument('--kernel_size', type=int, default=12)
    parser.add_argument('--nb_cnn_prot', type=int, default=5)
    parser.add_argument('--nb_cnn_cmpd', type=int, default=2)
    parser.add_argument('--padding', type=str, default='valid')

    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--dir_out', type=str, default='./results')
    parser.add_argument('--is_converge', type=bool, default=False)
    parser.add_argument('--verbose', type=int, default=1)

    parser.add_argument('--RE_epoch', type=int, default=0)
    args = parser.parse_args()

GPU_device = '/gpu:'+args.GPU

def set_callback():
    callback_list = []
    callback_list.append(task_val_call(model=model, path_w=ckpt_path, n_addL=0, path_csv=dir_ret, batch_test = 1024))

    return callback_list

def history_csv(history, dir_csv):
    key_list = history.history.keys()
    ret_list=[]
    for key in key_list:
        ret_list.append( history.history[key])

    pd.DataFrame(np.array(ret_list).T, columns=key_list).to_csv(dir_csv)



def eval_N_csv(y_oneHot, DB, csv_list, model, epoch_transfer, epoch_train, batch_size, X_in, path_csv, taskName=None, time_tr=None):
    y_oneHot = y_oneHot
    y_test_raw = categorical_probas_to_classes(y_oneHot)

    y_probas = model.predict(X_in, batch_size=batch_size, verbose=0)

    write_result( y_test_raw, y_probas, 
                 cvscores= csv_list, taskName=taskName, path = path_csv,
                 epoch_train=epoch_train, epoch_transfer=epoch_transfer, DB=DB, time_tr=time_tr)

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


def write_result(y_raw, y_probas, cvscores ,path ,DB, taskName='All', epoch_train=None, epoch_transfer=None, time_tr=0):
    y_pred = categorical_probas_to_classes(y_probas)
    y_target=y_raw[:len(y_pred)]

    roc_auc = roc_auc_score(y_target, y_probas[:,1])
    pre, rec, thresholds = precision_recall_curve(y_target, y_probas[:,1])
    prc = auc(rec, pre)

    acc, precision, npv, sensitivity, specificity, mcc, f1, tp, fp, tn, fn= calculate_performace(len(y_pred), y_target, y_pred)
    cvscores.append([acc, precision,npv, sensitivity, specificity, mcc,roc_auc, prc, 
                    tp, fp, tn, fn, 
                    DB,taskName, epoch_train, epoch_transfer, time_tr, model.count_params()] )
    
    cv_df = pd.DataFrame(cvscores, columns=['acc', 'precision','npv', 'sensitivity', 'specificity', 'mcc','roc_auc', 'prc',
                                            'true pos','false pos','true neg','false neg', 
                                            'DB','taskName','epoch_train', 'epoch_transfer', 'time_tr' ,'n_para'])

    cv_df.to_csv(path+'{DB}.csv'.format(DB=DB))

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
        path = base_path+"/Trans_{task}_on_train_epoch{epoch_tr}_addL{addL}/".format(task=task, 
        																			epoch_tr=epoch_tr,
        																			addL=addL)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# ## Change Keras Backend as TF
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

def data_valid(pair_dataset, 
				table_aa=None, table_prot0=None, 
				table_smile=None, table_mol2vec=None, table_mol2ecfp=None):
    x_list = []
    cmpd_list=[]

    # get protein 
    temp_prot=pair_dataset['protein'].values
    if args.PROT_type == 'V' or args.PROT_type == 'H':
        X_prot_vec = table_prot0.loc[temp_prot].values
        x_list.append( X_prot_vec)

    if args.PROT_type == 'R' or args.PROT_type == 'H':
        X_prot_aa = table_aa.loc[temp_prot].values
        x_list.append( X_prot_aa)

    # get compound
    temp_com = pair_dataset['mol_id'].values  
    if args.CMPD_type == 'V' or args.CMPD_type == 'H' and args.ecfp==1 :
        X_comp = table_mol2ecfp.loc[temp_com].values
        cmpd_list.append( X_comp)
    if args.CMPD_type == 'V' or args.CMPD_type == 'H' and args.vec==1:
        X_comp = table_mol2vec.loc[temp_com].values
        cmpd_list.append( X_comp)
    if args.CMPD_type == 'V' or args.CMPD_type == 'H' and args.ecfp==1 and args.vec==1:
        X_comp = np.concatenate( cmpd_list, axis=1) 

    if args.CMPD_type == 'V' or args.CMPD_type == 'H' and args.ecfp==1 or args.vec==1:
        x_list.append( X_comp)

    if args.CMPD_type == 'R' or args.CMPD_type == 'H':
        X_cmpd_smile = table_smile.loc[temp_com].values
        x_list.append( X_cmpd_smile)

    X_in= x_list
    y_oneHot = np_utils.to_categorical(pair_dataset['label'].values)
    return [X_in, y_oneHot]

def train_generator(batch_size, pair_dataset, table_aa=None, table_prot0=None, 
                    table_smile=None, table_mol2vec=None, table_mol2ecfp=None ):
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

            x_list = []
            cmpd_list=[]
            # get protein 
            temp_prot=pair_dataset['protein'].values[idx_batch]



            if args.PROT_type == 'V' or args.PROT_type == 'H':
                X_prot_vec = table_prot0.loc[temp_prot].values
                x_list.append( X_prot_vec)

            if args.PROT_type == 'R' or args.PROT_type == 'H':
                X_prot_aa = table_aa.loc[temp_prot].values
                x_list.append( X_prot_aa)


            # get compound
            temp_com = pair_dataset['mol_id'].values[idx_batch]

            if args.CMPD_type == 'V' or args.CMPD_type == 'H' and args.ecfp==1 :
                X_comp = table_mol2ecfp.loc[temp_com].values
                cmpd_list.append( X_comp)
            if args.CMPD_type == 'V' or args.CMPD_type == 'H' and args.vec==1:
                X_comp = table_mol2vec.loc[temp_com].values
                cmpd_list.append( X_comp)
            if args.CMPD_type == 'V' or args.CMPD_type == 'H' and args.ecfp==1 and args.vec==1:
                X_comp = np.concatenate( cmpd_list, axis=1) 

            if args.CMPD_type == 'V' or args.CMPD_type == 'H' and args.ecfp==1 or args.vec==1:
                x_list.append( X_comp)

            if args.CMPD_type == 'R' or args.CMPD_type == 'H':
                X_cmpd_smile = table_smile.loc[temp_com].values
                x_list.append( X_cmpd_smile)

            X_in= x_list
            yield X_in, y_oneHot[idx_batch]

def get_mol_name(ecfp, vec):
    if vec * ecfp == 1:
        mol_name = 'concat'
    elif vec ==1:
        mol_name = 'vec'
    elif ecfp ==1:
        mol_name = 'ecfp'
    else:
        mol_name = 'smiles'
    return mol_name

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
        self.path_csv = self.path_csv+'/'+fName+'epoch{epoch}_'
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

    def on_epoch_begin(self, epoch, logs=None, verbose=False):
        self.time_start=time.time()

        # if verbose is True:        
        #     str_model_info = para_info+'\ttask_val_on_epoch_begin@{}\tbased on trainE[{}/{:3d}]\t'.format(self.full_learning+'task',self.epoch_train, args.TRAIN_epoch)
        #     str_trainable_w= 'trainable weights:[{}]\t'.format(len(self.model_tr.trainable_weights))+'+'*len(self.model_tr.trainable_weights)
        #     print str_model_info, str_trainable_w

    def on_epoch_end(self, epoch, logs=None):
        time_tr = time.time()- self.time_start
        # 1.4. Evaluate task_te
        eval_N_csv(y_oneHot= task_val_y_oneHot, X_in= task_val_X_in, model=self.model, path_csv=self.path_csv.format(epoch=epoch),
                   epoch_train=self.epoch_train, epoch_transfer=epoch, batch_size=self.batch, 
                   csv_list=self.CSV_task_te, DB='task', taskName='task_te', time_tr=time_tr)
        
        # 2. Save Weights
        if self.path is not None and epoch%50==0 :
            # self.model_tr.save_weights(self.path_save.format(epoch=epoch), overwrite=True)
            self.model_tr.save(self.path_save.format(epoch=epoch), overwrite=True)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if args.gpu_frac < 1:
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
sess = tf.Session(config=config)

back_end='tensorflow'
set_keras_backend(back_end)
K.set_session(sess)
print( 'Backend is changed: '+ K.backend())


# 2.1 Get task pair data
task_fname_tr = '../../data/training_task/tr_pair_train_median.csv'
task_pair_train = pd.read_csv(task_fname_tr, index_col=0, header=0)

task_fname_val = '../../data/training_task/tr_pair_test_median.csv'
task_pair_val = pd.read_csv(task_fname_val, index_col=0, header=0)

# 2.2 Get task data Table
table_mol2vec  = pd.read_csv('../../data/training_task/tr_table_molregno-mol2vec_avg.csv', index_col=0)
table_mol2ecfp  = pd.read_csv('../../data/training_task/tr_table_molregno-ecfp.csv', index_col=0)
table_prot0  = pd.read_csv('../../data/training_task/tr_table_protVec_tf-idf0.csv', index_col=0)

scaler_mol2vec  = StandardScaler().fit(table_mol2vec)
scaler_mol2ecfp = StandardScaler().fit(table_mol2ecfp)
scaler_prot     = StandardScaler().fit(table_prot0)

# 2.3 Normalizing the Values
table_mol2vec   = scaled_df( scaler= scaler_mol2vec, df= table_mol2vec)
table_mol2ecfp  = scaled_df( scaler= scaler_mol2ecfp, df= table_mol2ecfp)
table_prot0     = scaled_df( scaler=scaler_prot, df= table_prot0)

table_aa    = pd.read_csv('../../data/training_task/tr_table_AA[700].csv', index_col=0)
table_smile = pd.read_csv('../../data/training_task/tr_table_smile[100].csv', index_col=0)

task_val_X_in, task_val_y_oneHot = data_valid(
                                       pair_dataset=task_pair_val,
                                       table_prot0= table_prot0,
                                       table_mol2vec= table_mol2vec, 
                                       table_mol2ecfp= table_mol2ecfp,
                                       table_aa=    table_aa,
                                       table_smile= table_smile,
                                        )



verbose = 0
steps_task = int(len(task_pair_train)//(args.TRAIN_batch))+1
if args.is_converge:
    callback_list = set_callback()
else:
    callback_list = None

fname_temp=__file__
mol_name = get_mol_name(ecfp = args.ecfp, vec = args.vec)
model_info = '_{}[{}-{}]f{}{}'.format(mol_name, args.SL_length, args.kernel_size, args.nb_filters, args.comment)

fName=fname_temp[:-3] + model_info

dir_model='./saved_models/{}/'.format(fName)
dir_model=set_path(dir_model)

dir_ret = args.dir_out+'/{}/'.format(fName)
dir_ret = set_path(dir_ret)
path_train = dir_ret+'train_model_history.csv'


########### Model Build! ##############
with tf.device('/cpu:0'):
    model= m_.get_PINN( PROT_type=args.PROT_type, 
    					CMPD_type=args.CMPD_type, 
    					SL_length=args.SL_length, 
    					CL_length=args.CL_length,

    					dropout_value=0.5, 
						num_filters = args.nb_filters, 
						embed_dim=args.embed_dim,
						
                        kernel_size=args.kernel_size,
                        nb_cnn_cmpd=args.nb_cnn_cmpd,
                        nb_cnn_prot=args.nb_cnn_prot,
                        padding = args.padding,

                        ecfp = args.ecfp,
                        vec = args.vec,
                        n_nodes = args.n_nodes)

if args.RE_epoch >0:
    re_path = dir_model+'model_tr_epoch{}.h5'.format(args.RE_epoch)
    model = load_model(re_path)
    print('re-load the model!')

with tf.device(GPU_device):
    parallel_model = multi_gpu_model(model, gpus=args.n_gpu)
    sgd = Adam( lr = args.TRAIN_lr, decay=1e-6)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', _mcc])

    ckpt_path = dir_model

        
    history = parallel_model.fit_generator(
                                 generator=train_generator(
                                                batch_size= args.TRAIN_batch,
                                                pair_dataset=task_pair_train,
                                                table_prot0=  table_prot0,
                                                table_mol2vec= table_mol2vec,
                                                table_mol2ecfp= table_mol2ecfp,
                                                table_aa = table_aa,
                                                table_smile = table_smile,
                                                ),
                                  steps_per_epoch= steps_task,
                                  epochs=args.TRAIN_epoch,
                                  verbose= args.verbose,
                                  callbacks= callback_list,
                                  max_queue_size=8 ,
                                  workers=4,
                                  use_multiprocessing=False, 
                                  shuffle=True,
                                  initial_epoch= args.RE_epoch,
                                  )
    history_csv(history, path_train)
