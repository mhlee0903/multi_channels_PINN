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
import models_elu_feat as m_
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
    parser.add_argument('--TRAIN_epoch', type=int, default=100)
    parser.add_argument('--TRAIN_lr', type=float, default=0.0005)


    # parser.add_argument('--ecfp', 		type=int, default=1)
    # parser.add_argument('--vec', 		type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--nb_filters', type=int, default=64)

    parser.add_argument('--isTest', type=bool, default=False)
    
    parser.add_argument('--TRANS_batch', type=int, default=1024)
    parser.add_argument('--TRANS_epoch', type=int, default=151)
    parser.add_argument('--TRANS_lr', type=float, default=0.0005)
    parser.add_argument('--TRANS_n_addL', type=int, default=0)
    parser.add_argument('--TRANS_opt', type=str, default='hype_full')
    parser.add_argument('--Tran_start_epoch', type=int, default=0)

    parser.add_argument('--RE_model', type=str, default='')
    parser.add_argument('--RE_epoch', type=int, default=0)

    parser.add_argument('--TAXONOMY', type=str, default='rat_all')
    parser.add_argument('--MODEL_str', type=str, default='3-1')
    args = parser.parse_args()

GPU_device = '/gpu:'+args.GPU

def set_callback(TRANS_opt):
    callback_list = []

    # callback_list.append(EarlyStopping(monitor='val__mcc', patience = nb_patience, verbose=1, mode='max'))
    callback_list.append(hype_val_call(model=model, path_w=ckpt_path, n_addL=0, path_csv=dir_ret, batch_test = 1024))

    if TRANS_opt !='':
        transfer_hype_half = transfer_(model=model, path_w =ckpt_path,  path_csv=dir_ret, task='hype', full_learning='HALF', n_addL=args.TRANS_n_addL, lr=args.TRANS_lr, n_epoch=args.TRANS_epoch, batch_size=args.TRANS_batch)
        transfer_hype_full = transfer_(model=model, path_w =ckpt_path,  path_csv=dir_ret, task='hype', full_learning='FULL', n_addL=args.TRANS_n_addL, lr=args.TRANS_lr, n_epoch=args.TRANS_epoch, batch_size=args.TRANS_batch)
        transfer_hype_none = transfer_(model=model, path_w =ckpt_path,  path_csv=dir_ret, task='hype',                       n_addL=args.TRANS_n_addL, lr=args.TRANS_lr, n_epoch=args.TRANS_epoch, batch_size=args.TRANS_batch)

    if TRANS_opt=='all':
        callback_list.append( transfer_hype_half)
        callback_list.append( transfer_hype_full)
        callback_list.append( transfer_hype_none)

    if TRANS_opt=='hype_all':
        callback_list.append( transfer_hype_half)
        callback_list.append( transfer_hype_full)
        callback_list.append( transfer_hype_none)

    if TRANS_opt=='hype':
        callback_list.append( transfer_hype_half)
        callback_list.append( transfer_hype_full)

    if TRANS_opt=='hype_full':
        callback_list.append( transfer_hype_full)

    if TRANS_opt=='hype_half':
        callback_list.append( transfer_hype_half)
        
    if TRANS_opt=='':
        print('NOT Transfered..')
   
    # print('TRANS_opt: {}\t Callback#: '.format(args.TRANS_opt, len(callback_list)-1))
    # print('-'*30)
    return callback_list

def history_csv(history, dir_csv):
    key_list = history.history.keys()
    ret_list=[]
    for key in key_list:
        ret_list.append( history.history[key])

    pd.DataFrame(np.array(ret_list).T, columns=key_list).to_csv(dir_csv)


def trainable_Model(model, isTrain, addL=0, half=False):
    for layer in model.layers :
        layer.trainable = True

    if not isTrain :
        freezeL = -(addL*2+1)
        if half:
            half_layers = int(1.0*(len(model.layers)-(addL*2+1))/2)
            freezeL -= half_layers

        for layer in model.layers[:freezeL] :
            layer.trainable = isTrain

def new_Model(model_tr, addL=0, n_nodes=128):
    init = 'lecun_normal'
    active_func = 'selu'
    
    input_prot = model_tr.layers[0].input
    input_cmpd = model_tr.layers[1].input
    output = model_tr.layers[-2].output
    
    if addL >0:
        for n_layer in range(addL):
            output=Dense(n_nodes, activation=active_func, kernel_initializer=init, name='new_layer'+str(n_layer))(output)
            # output=AlphaDropout(0.1)(output)
            output=Dropout(0.5)(output)
    output=Dense(2, activation='softmax', name='new_output')(output)

    return Model( input=[input_prot,input_cmpd], output=output)


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

def data_valid(pair_dataset, table_prot0, table_mol2vec, table_mol2ecfp):
    # get protein 
    temp_prot=pair_dataset['protein'].values  
    X_prot_vec = table_prot0.loc[temp_prot].values

    # get compound
    temp_com = pair_dataset['mol_id'].values  
    X_cmpd_vec = table_mol2vec.loc[temp_com].values
    X_cmpd_ecfp = table_mol2ecfp.loc[temp_com].values
    X_comp = np.concatenate( (X_cmpd_ecfp, X_cmpd_vec ), axis=1) 

    X_in=[X_prot_vec, X_comp ]
    y_oneHot = np_utils.to_categorical(pair_dataset['label'].values)
    return [X_in, y_oneHot]


def train_generator(batch_size, pair_dataset, table_prot0, table_mol2vec, table_mol2ecfp):
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
            X_prot_vec = table_prot0.loc[temp_prot].values
            
            # get compound
            temp_com = pair_dataset['mol_id'].values[idx_batch]  
            X_cmpd_vec = table_mol2vec.loc[temp_com].values
            X_cmpd_ecfp = table_mol2ecfp.loc[temp_com].values
            X_cmpd = np.concatenate( (X_cmpd_ecfp, X_cmpd_vec ), axis=1) 
            

            X_in=[X_prot_vec,  X_cmpd]
            yield X_in, y_oneHot[idx_batch]




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

back_end='tensorflow'
set_keras_backend(back_end)
K.set_session(sess)
print( 'Backend is changed: '+ K.backend())

###################################################

verbose = 1

MODEL_dic = {'4-1':m_.pcm4_1, '3-1':m_.pcm3_1, '2-1':m_.pcm2_1}


# 1.1 Hype Pair
if args.TAXONOMY=='rat_high':
    trans_pair_tr  = pd.read_csv('../../data/transfer_data/trasfer[rat_mus]_conf[9]_black[C].csv', index_col=0)

elif args.TAXONOMY=='rat_all':
    trans_pair_tr  = pd.read_csv('../../data/transfer_data/trasfer[rat_mus]_conf[9low]_black[C].csv', index_col=0)
else:
    print("There is NO INPUT!!!")

# 2.1 Get Hype pair data
task_fname_val ='../../data/task_data/task_pair_test.csv'
task_pair_val = pd.read_csv(hype_fname_val, index_col=0, header=0)
task_fname_tr ='../../data/task_data/task_pair_train.csv'
task_pair_train = pd.read_csv(hype_fname_trans, index_col=0, header=0)

# 2.2 Get Hype data Table
table_mol2vec  = pd.read_csv('../../data/transfer_data/transfer_table_mol2vec.csv', index_col=0)
table_mol2ecfp = pd.read_csv('../../data/transfer_data/transfer_table_ecfp.csv', index_col=0)
table_prot0    = pd.read_csv('../../data/transfer_data/transfer_table_protVec.csv', index_col=0)

scaler_mol2vec = StandardScaler().fit(table_mol2vec)
scaler_mol2ecfp = StandardScaler().fit(table_mol2ecfp)
scaler_prot = StandardScaler().fit(table_prot0)


# 2.3 Normalizing the Values
table_mol2vec = scaled_df( scaler= scaler_mol2vec, df= table_mol2vec)
table_mol2ecfp = scaled_df( scaler= scaler_mol2ecfp, df= table_mol2ecfp)
table_prot0 = scaled_df( scaler=scaler_prot, df= table_prot0)

task_val_X_in, task_val_y_oneHot = data_valid(
                                       pair_dataset=task_pair,
                                       table_prot0= table_prot0,
                                       table_mol2vec= table_mol2vec, 
                                       table_mol2ecfp= table_mol2ecfp )

class hype_val_call(Callback):
    def __init__(self, model, path_csv,  n_addL, epoch_train=None, path_w= None, full_learning='', batch_test= 1024):        
        # set the CSV list
        self.CSV_hype_te =[]
        
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

    def on_epoch_begin(self, epoch, logs=None, verbose=False):
        # if verbose is True:        
        #     str_model_info = para_info+'\thype_val_on_epoch_begin@{}\tbased on trainE[{}/{:3d}]\t'.format(self.full_learning+'Hype',self.epoch_train, args.TRAIN_epoch)
        #     str_trainable_w= 'trainable weights:[{}]\t'.format(len(self.model_tr.trainable_weights))+'+'*len(self.model_tr.trainable_weights)
        #     print str_model_info, str_trainable_w

    def on_epoch_end(self, epoch, logs=None):
        # 1.4. Evaluate hype_te
        eval_N_csv(y_oneHot= task_val_y_oneHot, X_in= task_val_X_in, model=self.model, path_csv=self.path_csv.format(epoch=epoch),
                   epoch_train=self.epoch_train, epoch_transfer=epoch, batch_size=self.batch, 
                   csv_list=self.CSV_hype_te, DB='Hype', taskName='hype_te' )
        
        # 2. Save Weights
        if self.path is not None :
            # self.model_tr.save_weights(self.path_save.format(epoch=epoch), overwrite=True)
            self.model_tr.save(self.path_save.format(epoch=epoch), overwrite=True)

class transfer_(Callback):
    def __init__(self, model, path_w, path_csv, task, full_learning='',
                 n_addL=0, lr=0.0005, n_epoch=10, batch_size=1024 ):        
        
        # set the CSV list
        self.CSV__tr =[]
        # We set the model (non multi gpu) under an other name
        self.model_tr = model
        with tf.device('/cpu:0'):
            self.model_ = MODEL_dic[args.MODEL_str](dropout_value=0.5 )
            
        # hyper parameters
        self.n_addL = n_addL
        self.lr = lr
        self.n_epoch = n_epoch
        self.batch_size = batch_size

        # set path with task
        self.path_w_ =  path_w
        self.path_csv_ =  path_csv
        
        self.task = task
        self.full_learning=full_learning
    
    def on_epoch_begin(self, epoch, logs=None):
        trainable_Model(model= self.model_tr, isTrain=True)
        # str_model_info = para_info+'\ttransfer_on_epoch_begin@transfer_{}_\tbased on trainE[{}/{:3d}]'.format(self.task , epoch, args.TRAIN_epoch)
        # str_trainable_w= 'trainable weights:[{}]\t'.format(len(self.model_tr.trainable_weights))+'+'*len(self.model_tr.trainable_weights)
        # print str_model_info, str_trainable_w
        
    def on_epoch_end(self, epoch_train, logs=None):
        if (args.Tran_start_epoch <= epoch_train): # and (epoch_train %2 ==0 or epoch_train<10):
            # 1.copy W-> 2.freeze W-> 3.cut & new layer -> 4.lr setting -> 5. evaluate D_test as raw -> 6. Transfer Learning
            if args.isTest:
                self.n_epoch = 1


            # 1.copy weights
            if self.full_learning is 'FULL':
                self.model_.set_weights( self.model_tr.get_weights())
                trainable_Model(self.model_, isTrain=True)

            elif self.full_learning is 'HALF':
                self.model_.set_weights( self.model_tr.get_weights())
                trainable_Model(self.model_, isTrain=False, addL=self.n_addL, half=True)

            else:
                # Direct.ver (Not Copy, but SHARE)
                self.model_ = new_Model(model_tr=self.model_tr, addL=self.n_addL)
                # 2.freeze & 3.cut&new
                trainable_Model(self.model_, isTrain=False, addL=self.n_addL )


            # 4. lr setting
            with tf.device(GPU_device):
                parallel_ = multi_gpu_model(self.model_, gpus=args.n_gpu)
                parallel_.compile(loss='categorical_crossentropy', 
                                       optimizer=Adam( lr= self.lr , decay=1e-6), 
                                       metrics=['accuracy', _mcc])

                path_ckpt = None
                path_csv =  set_path(self.path_csv_+'/Trans_on_train')
                path_csv= path_csv+"/Trans_on_train_epoch{epoch_tr}_addL{addL}_".format( epoch_tr=epoch_train ,addL=self.n_addL)+self.full_learning
                path_train = path_csv+self.task+'_train.csv'

                # 5. Train the transfer model
                callback_list=[hype_val_call(
                                        model=self.model_, n_addL=self.n_addL, epoch_train=epoch_train, 
                                        path_w=path_ckpt, path_csv=path_csv, batch_test = 1024, full_learning=self.full_learning), 
                              # EarlyStopping(monitor='val__mcc', patience = patience_hype, verbose=1, mode='max'),
                              ]

                history = parallel_.fit_generator(
                                         generator=train_generator(
                                                        batch_size= self.batch_size ,
                                                        pair_dataset=task_pair_train,
                                                        table_prot0=  table_prot0,
                                                        table_mol2vec= table_mol2vec,
                                                        table_mol2ecfp= table_mol2ecfp)  ,
                                          steps_per_epoch= steps_task,
                                          epochs=self.n_epoch,
                                          callbacks=callback_list,
                                          max_queue_size=12,
                                          workers=2,
                                          use_multiprocessing=False, 
                                          shuffle=True,                                     
                                          verbose=1,
                                                    )
                history_csv(history, path_train)
                del parallel_

############################
steps_trans = int(len(trans_pair_tr)//(args.TRAIN_batch))+1
steps_task = int(len(task_pair_train)//(args.TRAIN_batch))+1
nb_patience = 100
patience_hype=int(nb_patience/2)
#############################

########### Model Info setting ##############
fname_temp=__file__
fName=fname_temp[:-3]

# layer_info = 'Tax[{}]_Model[{}]_'.format(args.TAXONOMY, args.MODEL_str)
# # para_info=layer_info+'Trans[Lr{}_addL{}]'.format(args.TRANS_lr, args.TRANS_n_addL)
# para_info=layer_info+'CNN[edim{}_filters{}]'.format(args.embed_dim, args.nb_filters)


time_now = datetime.datetime.now().strftime('%m-%d %H:%M')

dir_model='./saved_models/'+fName+'/'# +para_info+'/'
dir_model=set_path(dir_model)

dir_ret = './results/'+fName+'/'# +para_info+'/'
dir_ret=set_path(dir_ret)
path_train = dir_ret+'train_model_history.csv'

########### Model Build! ##############
with tf.device('/cpu:0'):
    model= MODEL_dic[args.MODEL_str](dropout_value=0.5 )

if args.RE_model is not '':
    re_path = dir_model+args.RE_model
    model = load_model(re_path)
    print('re-load the model!\t'+args.RE_model)

with tf.device(GPU_device):
    parallel_model = multi_gpu_model(model, gpus=args.n_gpu)
    sgd = Adam( lr = args.TRAIN_lr, decay=1e-6)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', _mcc])

    # Setting the callback functions
    ckpt_path = dir_model
    callback_list = set_callback(args.TRANS_opt)
        
    history = parallel_model.fit_generator(
                                 generator=train_generator(
                                                batch_size= 256*4,
                                                pair_dataset=hype_pair_tr,
                                                table_prot0=  table_prot0,
                                                table_mol2vec= table_mol2vec,
                                                table_mol2ecfp= table_mol2ecfp)  ,
                                  steps_per_epoch= steps_trans,
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
