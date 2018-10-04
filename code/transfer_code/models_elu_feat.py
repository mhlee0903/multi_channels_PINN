from keras.models import Model
from keras.layers import Dense, concatenate,Input, AlphaDropout, Dropout, Embedding, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization, Activation
from keras.layers.merge import concatenate


def pcm2_1(dropout_value=0.5, ecfp=1, vec=1 ):
    init = 'lecun_normal'
    CNN_act ='elu'
    active_func ='elu'

    prot_vec = Input(shape=(300,), name='prot_vec')
    mol_vec  = Input(shape=(1024*ecfp +300*vec ,), name='mol_vec')

    _fc_protein = Dense(2048, activation=active_func, kernel_initializer=init, name='Prot_feature1')(prot_vec)
    _fc_protein=Dropout(0.1)(_fc_protein)
    _fc_protein = Dense(512, activation=active_func, kernel_initializer=init, name='Prot_feature3')(_fc_protein)
    _fc_protein=Dropout(dropout_value)(_fc_protein)

    _fc_cmpd = Dense(2048,  activation=active_func, kernel_initializer=init, name='Cmpd_feature1')(mol_vec)
    _fc_cmpd=Dropout(0.1)(_fc_cmpd)
    _fc_cmpd = Dense(512, activation=active_func, kernel_initializer=init, name='Cmpd_feature3')(_fc_cmpd)
    _fc_cmpd=Dropout(dropout_value)(_fc_cmpd)


    merged_vector = concatenate([_fc_protein, _fc_cmpd, ], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature_1')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[ prot_vec, mol_vec ], output=outputs)
    return model

def pcm3_1(dropout_value=0.5, ecfp=1, vec=1 ):
    init = 'lecun_normal'
    CNN_act ='elu'
    active_func ='elu'

    prot_vec = Input(shape=(300,), name='prot_vec')
    mol_vec  = Input(shape=(1024*ecfp +300*vec ,), name='mol_vec')

    _fc_protein = Dense(2048, activation=active_func, kernel_initializer=init, name='Prot_feature1')(prot_vec)
    _fc_protein=Dropout(0.1)(_fc_protein)
    _fc_protein = Dense(1024, activation=active_func, kernel_initializer=init, name='Prot_feature2')(_fc_protein)
    _fc_protein=Dropout(dropout_value)(_fc_protein)
    _fc_protein = Dense(256, activation=active_func, kernel_initializer=init, name='Prot_feature5')(_fc_protein)
    _fc_protein=Dropout(dropout_value)(_fc_protein)

    _fc_cmpd = Dense(2048,  activation=active_func, kernel_initializer=init, name='Cmpd_feature1')(mol_vec)
    _fc_cmpd=Dropout(0.1)(_fc_cmpd)
    _fc_cmpd = Dense(1024, activation=active_func, kernel_initializer=init, name='Cmpd_feature2')(_fc_cmpd)
    _fc_cmpd=Dropout(dropout_value)(_fc_cmpd)
    _fc_cmpd = Dense(256, activation=active_func, kernel_initializer=init, name='Cmpd_feature5')(_fc_cmpd)
    _fc_cmpd=Dropout(dropout_value)(_fc_cmpd)


    merged_vector = concatenate([_fc_protein, _fc_cmpd, ], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature_1')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[ prot_vec, mol_vec ], output=outputs)
    return model

def pcm4_1(dropout_value=0.5, ecfp=1, vec=1 ):
    init = 'lecun_normal'
    CNN_act ='selu'
    active_func ='selu'

    prot_vec = Input(shape=(300,), name='prot_vec')
    mol_vec  = Input(shape=(1024*ecfp +300*vec ,), name='mol_vec')

    _fc_protein = Dense(2048, activation=active_func, kernel_initializer=init, name='Prot_feature1')(prot_vec)
    _fc_protein=Dropout(0.1)(_fc_protein)
    _fc_protein = Dense(1024, activation=active_func, kernel_initializer=init, name='Prot_feature2')(_fc_protein)
    _fc_protein=Dropout(dropout_value)(_fc_protein)
    _fc_protein = Dense(512, activation=active_func, kernel_initializer=init, name='Prot_feature3')(_fc_protein)
    _fc_protein=Dropout(dropout_value)(_fc_protein)
    _fc_protein = Dense(256, activation=active_func, kernel_initializer=init, name='Prot_feature5')(_fc_protein)
    _fc_protein=Dropout(dropout_value)(_fc_protein)

    _fc_cmpd = Dense(2048,  activation=active_func, kernel_initializer=init, name='Cmpd_feature1')(mol_vec)
    _fc_cmpd=Dropout(0.1)(_fc_cmpd)
    _fc_cmpd = Dense(1024, activation=active_func, kernel_initializer=init, name='Cmpd_feature2')(_fc_cmpd)
    _fc_cmpd=Dropout(dropout_value)(_fc_cmpd)
    _fc_cmpd = Dense(512, activation=active_func, kernel_initializer=init, name='Cmpd_feature3')(_fc_cmpd)
    _fc_cmpd=Dropout(dropout_value)(_fc_cmpd)
    _fc_cmpd = Dense(256, activation=active_func, kernel_initializer=init, name='Cmpd_feature5')(_fc_cmpd)
    _fc_cmpd=Dropout(dropout_value)(_fc_cmpd)


    merged_vector = concatenate([_fc_protein, _fc_cmpd, ], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature_1')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[ prot_vec, mol_vec ], output=outputs)
    return model