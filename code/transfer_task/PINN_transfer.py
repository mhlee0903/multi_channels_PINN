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

    parser.add_argument('--isTest', type=bool, default=False)
    parser.add_argument('--TRANS_batch', type=int, default=1024)
    parser.add_argument('--TRANS_epoch', type=int, default=201)
    parser.add_argument('--TRANS_lr', type=float, default=0.0005)
    parser.add_argument('--n_detail', type=int, default=150)
    parser.add_argument('--tox_patience', type=int, default=40)
    parser.add_argument('--sample_w', type=bool, default=True)

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
    parser.add_argument('--Tran_start_epoch', type=int, default=0)

    args = parser.parse_args()

GPU_device = '/gpu:'+args.GPU
csv_save_term = args.TRANS_epoch //2

def set_callback():
    callback_list = []
    callback_list.append(task_val_call(model=model, path_w=ckpt_path, n_addL=0, path_csv=dir_ret, batch_test = 1024))
    callback_list.append(transfer_(model=model, path_w=None, path_csv=dir_ret, task='tox', full_learning='FULL',
                                     n_addL=0, lr=args.TRANS_lr, n_epoch=args.TRANS_epoch, batch_size=args.TRANS_batch ,
                                     comment='_1'
                                  ))
    callback_list.append(transfer_(model=model, path_w=None, path_csv=dir_ret, task='tox', full_learning='HALF',
                                     n_addL=0, lr=args.TRANS_lr, n_epoch=args.TRANS_epoch, batch_size=args.TRANS_batch,
                                     comment='_1'
                                  ))

    callback_list.append(transfer_(model=model, path_w=None, path_csv=dir_ret, task='tox', full_learning='FULL',
                                     n_addL=0, lr=args.TRANS_lr, n_epoch=args.TRANS_epoch, batch_size=args.TRANS_batch ,
                                     comment='_2'
                                  ))
    callback_list.append(transfer_(model=model, path_w=None, path_csv=dir_ret, task='tox', full_learning='HALF',
                                     n_addL=0, lr=args.TRANS_lr, n_epoch=args.TRANS_epoch, batch_size=args.TRANS_batch,
                                     comment='_2'
                                  ))
    return callback_list

def history_csv(history, dir_csv):
    key_list = history.history.keys()
    ret_list=[]
    for key in key_list:
        ret_list.append( history.history[key])

    pd.DataFrame(np.array(ret_list).T, columns=key_list).to_csv(dir_csv)



def eval_N_csv(y_oneHot, DB, csv_list, model, epoch_transfer, epoch_train, batch_size, X_in, path_csv, taskName=None, is_force=False):
    y_oneHot = y_oneHot
    y_test_raw = categorical_probas_to_classes(y_oneHot)

    y_probas = model.predict(X_in, batch_size=batch_size, verbose=0)



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



def data_valid(pair_dataset, 
				table_aa=None, table_prot0=None, 
				table_smile=None, table_mol2vec=None, table_mol2ecfp=None, is_w=False):
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

    if is_w:
        w = pair_dataset['w']
    else:
        w = None
    return [X_in, y_oneHot, w]


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

def new_Model(model_tr, addL=0, n_nodes=256):
    init = 'lecun_normal'
    active_func = 'selu'
    
#     prot_vec = model_tr.layers[0].input
#     input_cmpd = model_tr.layers[1].input
    output = model_tr.layers[-2].output
    
    if addL >0:
        for n_layer in range(addL):
            output=Dense(n_nodes, activation=active_func, kernel_initializer=init, name='new_layer'+str(n_layer))(output)
            # output=AlphaDropout(0.1)(output)
            output=Dropout(0.5)(output)
    output=Dense(2, activation='softmax', name='new_output')(output)

#     return Model( input=[input_prot,input_cmpd], output=output)
    return Model( input=model_tr.inputs, output=output)


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

    def on_epoch_end(self, epoch, logs=None):
        # 1.4. Evaluate task_te
        eval_N_csv(y_oneHot= task_val_y_oneHot, X_in= task_val_X_in, model=self.model, path_csv=self.path_csv.format(epoch=epoch),
                   epoch_train=self.epoch_train, epoch_transfer=epoch, batch_size=self.batch, 
                   csv_list=self.CSV_task_te, DB='task', taskName='task_te' )
        
        # 2. Save Weights
        if self.path is not None :
            self.model_tr.save(self.path_save.format(epoch=epoch), overwrite=True)


class tox_val_call(Callback):
    def __init__(self, model, path_csv,  n_addL, epoch_train=None, path_w= None, batch_test= 1024):        
        # set the CSV list
        self.CSV_list =[]
        for i in range(len( tox_test_in_list)):
        	self.CSV_list.append( [])
        self.CSV_test =[]
        
        # We set the model (non multi gpu) under an other name
        self.model_tr = model
        self.path_dir = path_csv
        # set path for csv file
        if epoch_train is not None:
            self.path_dir = path_csv+'/tasks/'
            self.path_dir_full = path_csv
        else:
        	self.path_dir = path_csv+'/val_direct/'
        if not os.path.exists(self.path_dir):
            os.makedirs(self.path_dir)     
        
        self.path_addL = '/add{n_addL}_'.format(n_addL=n_addL)
        self.path_csv = self.path_dir_full+self.path_addL+'epoch{epoch}_'
        
        self.path_csv_task = self.path_dir +self.path_addL+'epoch{epoch}_'

        self.batch = batch_test
        self.n_addL=n_addL
        self.path_w = path_w
        
        self.epoch_train=epoch_train
        self.epoch_now =None
    def on_train_begin(self,logs=None):
        print('epoch_tr:{}\tis_add:{}   \r'.format( self.epoch_train, self.n_addL)),

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_now=epoch
        idx_csv=0
        eval_N_csv(y_oneHot=tox_test_y_oneHot, X_in=tox_test_X_in, model=self.model, path_csv=self.path_csv.format(epoch=epoch),
                   epoch_transfer=epoch, epoch_train=self.epoch_train, batch_size=self.batch, 
                   csv_list=self.CSV_test, DB='Tox21', taskName='test_all')

    def on_train_end(self,logs=None):
        epoch = self.epoch_now
        eval_N_csv(y_oneHot=tox_test_y_oneHot, X_in=tox_test_X_in, model=self.model, path_csv=self.path_csv.format(epoch=epoch),
                   epoch_transfer=epoch, epoch_train=self.epoch_train, batch_size=self.batch, 
                   csv_list=self.CSV_test, DB='Tox21', taskName='test_all', is_force=True)


class transfer_(Callback):
    def __init__(self, model, path_w, path_csv, task , full_learning='',
                 n_addL=0, lr=0.0005, n_epoch=10, batch_size=1024, comment='' ):        
        
        # set the CSV list
        self.CSV__tr =[]
        # We set the model (non multi gpu) under an other name
        self.model_tr = model
        with tf.device('/cpu:0'):
            self.model_ = MODEL_dic[args.MODEL_str](dropout_value=0.5, 
									num_filters = args.nb_filters, 
									embed_dim=args.embed_dim,
									kernel_size=args.kernel_size,
                                    nb_cnn=args.nb_cnn,
                                    ecfp = args.ecfp,
                                    vec = args.vec)
        # hyper parameters
        self.n_addL = n_addL
        self.lr = lr
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        # if task is 'tox':
        #     self.lr = lr*0.5

        # set path with task
        self.path_w_ =  path_w
        self.path_csv_ =  path_csv
        
        self.task = task
        self.full_learning=full_learning
        self.comment = comment

    def on_epoch_begin(self, epoch, logs=None):
        trainable_Model(model= self.model_tr, isTrain=True)
        
    def on_epoch_end(self, epoch_train, logs=None):
        if (args.Tran_start_epoch <= epoch_train) and epoch_train<args.n_detail or epoch_train%5==0:
            # 1.copy W-> 2.freeze W-> 3.cut & new layer -> 4.lr setting -> 5. evaluate D_test as raw -> 6. Transfer Learning
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
                                       optimizer=Adam( lr= self.lr , decay=1e-4), 
                                       metrics=['accuracy', _mcc])

                path_ckpt = None
                trans_info = '/Trans_add{addL}_{full}{comment}'.format( addL=self.n_addL, full=self.full_learning, comment=self.comment)
                path_csv =  set_path(self.path_csv_+trans_info)
                path_csv= path_csv+"/Trans_on_{epoch_tr}".format( epoch_tr=epoch_train)
                path_train_history = path_csv+self.task+'_train.csv'
                
                
                if args.sample_w:
                    sample_w = tox_train_w
                else:
                    sample_w = None
                
                # 5. Train the transfer model
                if self.task is 'tox':
                    callback_list=[tox_val_call(
                                            model=self.model_, n_addL=self.n_addL, epoch_train=epoch_train, 
                                            path_w=path_ckpt, path_csv=path_csv, batch_test = 1024), 
                                  EarlyStopping(monitor='val__mcc', patience = args.tox_patience, verbose=1, mode='max'),
                                  ]

                    history = parallel_.fit(x=tox_train_X_in, y=tox_train_y_oneHot, batch_size=self.batch_size, epochs=self.n_epoch,
                                            sample_weight= sample_w, 
                                             callbacks=callback_list, verbose=verbose,
                                             validation_data= (tox_val_X_in, tox_val_y_oneHot)
                                             )

                history_csv(history, path_train_history)
                del parallel_



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



tox_tasks= ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
       'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
        'SR-HSE','SR-MMP', 'SR-p53']

tox_pair_train = pd.read_csv('../../data/pairs/Tox21_pair_train.csv')
tox_pair_val = pd.read_csv('../../data/pairs/Tox21_pair_val.csv')
tox_pair_test = pd.read_csv('../../data/pairs/Tox21_pair_test.csv')

tox_task_list = []
for taskName in tox_tasks:
    df = pd.read_csv('../../data/pairs/Tox21_pair_test_{prot}.csv'.format(prot=taskName), index_col=0)
    tox_task_list.append(df)

tox_table_mol2vec = pd.read_csv('../../data/Tox21_table_smile2vec_avg.csv', index_col=0)
tox_table_mol2ecfp= pd.read_csv('../../data/Tox21_table_smile2ecfp.csv', index_col=0)
tox_table_prot    = pd.read_csv('../../data/Tox21_table_tf-idf_prot0[size300_sg1_window35_minCount2].csv', index_col=0)

tox_table_mol2vec   = scaled_df( scaler= scaler_mol2vec, df= tox_table_mol2vec)
tox_table_mol2ecfp  = scaled_df( scaler= scaler_mol2ecfp, df= tox_table_mol2ecfp)
tox_table_prot     = scaled_df( scaler=scaler_prot, df= tox_table_prot)

tox_table_aa = pd.read_csv('../../data/Tox21_table_AA700.csv', index_col=0)
tox_table_smile= pd.read_csv('../../data/Tox21_table_smile100_smile.csv', index_col=0)

tox_train_X_in, tox_train_y_oneHot, tox_train_w = data_valid(
                                       pair_dataset=tox_pair_train,
                                       table_prot0= tox_table_prot,
                                       table_mol2vec= tox_table_mol2vec, 
                                       table_mol2ecfp= tox_table_mol2ecfp,
                                       table_aa=    tox_table_aa,
                                       table_smile= tox_table_smile,
                                        is_w=True
                                        )

tox_val_X_in, tox_val_y_oneHot,_ = data_valid(
                                       pair_dataset=tox_pair_val,
                                       table_prot0= tox_table_prot,
                                       table_mol2vec= tox_table_mol2vec, 
                                       table_mol2ecfp= tox_table_mol2ecfp,
                                       table_aa=    tox_table_aa,
                                       table_smile= tox_table_smile,
                                        )

tox_test_X_in, tox_test_y_oneHot,_ = data_valid(
                                       pair_dataset=tox_pair_test,
                                       table_prot0= tox_table_prot,
                                       table_mol2vec= tox_table_mol2vec, 
                                       table_mol2ecfp= tox_table_mol2ecfp,
                                       table_aa=    tox_table_aa,
                                       table_smile= tox_table_smile,
                                        )

tox_test_in_list = []
for task_name, tox_test in zip(tox_tasks , tox_task_list):
    test_X_in, test_y_oneHot,_ = data_valid(
                                           pair_dataset=tox_test,
                                           table_prot0= tox_table_prot,
                                           table_mol2vec= tox_table_mol2vec, 
                                           table_mol2ecfp= tox_table_mol2ecfp,
                                           table_aa=    tox_table_aa,
                                           table_smile= tox_table_smile,
                                            )
    
    tox_test_in_list.append([task_name, test_X_in, test_y_oneHot])



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
ckpt_path = dir_model

with tf.device(GPU_device):
    parallel_model = multi_gpu_model(model, gpus=args.n_gpu)
    sgd = Adam( lr = args.TRAIN_lr, decay=1e-6)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', _mcc])
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

