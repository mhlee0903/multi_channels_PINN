from keras.models import Model
from keras.layers import Dense, concatenate,Input, AlphaDropout, Dropout, Embedding, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization, Activation
from keras.layers.merge import concatenate


UNIQUE_PROT= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
MAXLEN_PROT= 700

# UNIQUE_CMPD= ['#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4',
#        '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'F', 'G', 'H',
#        'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'V', 'W', 'Z',
#        '[', '\\', ']', 'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'n', 'o','r', 's', 't', 'u']
MAXLEN_CMPD= 100
UNIQUE_CMPD = ['#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4',
       '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'F', 'G', 'H',
       'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'V', 'W', 'Z',
       '[', '\\', ']', 'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'n', 'o','p', 'r', 's', 't', 'u']



def pcm2_1(dropout_value=0.5, ecfp=1, vec=1, num_filters = 32, embed_dim=16, kernel_size=4 ):
    init = 'lecun_normal'
    CNN_act ='elu'
    active_func ='elu'

    prot_vec = Input(shape=(300,), name='prot_vec')
    mol_vec  = Input(shape=(1024*ecfp +300*vec ,), name='mol_vec')
    prot_char_in  = Input(shape=(700,), name='prot_char_in')
    cmpd_smile_in = Input(shape=(100,), name='cmpd_smile_in')

    _fc_protein = Dense(1024, activation=active_func, kernel_initializer=init, name='Prot_feature1')(prot_vec)
    _fc_protein=Dropout(0.1)(_fc_protein)
    _fc_protein = Dense(256, activation=active_func, kernel_initializer=init, name='Prot_feature5')(_fc_protein)
    _fc_protein=Dropout(dropout_value)(_fc_protein)

    _fc_cmpd = Dense(2048,  activation=active_func, kernel_initializer=init, name='Cmpd_feature1')(mol_vec)
    _fc_cmpd=Dropout(0.1)(_fc_cmpd)
    _fc_cmpd = Dense(256, activation=active_func, kernel_initializer=init, name='Cmpd_feature5')(_fc_cmpd)
    _fc_cmpd=Dropout(dropout_value)(_fc_cmpd)
############

    prot_embedd = Embedding(input_dim=len(UNIQUE_PROT), output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_PROT)(prot_char_in)
    cmpd_embedd = Embedding(input_dim=len(UNIQUE_CMPD)+1, output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_CMPD)(cmpd_smile_in)

    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=1 , kernel_initializer=init)(prot_embedd)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=2 , kernel_initializer=init)(prot_i)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=4 , kernel_initializer=init)(prot_i)

    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=1 , kernel_initializer=init)(cmpd_embedd)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=2 , kernel_initializer=init)(cmpd_i)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=4 , kernel_initializer=init)(cmpd_i)

    _cnn_protein= Flatten()(prot_i)
    _cnn_cmpd= Flatten()(cmpd_i)

    merged_vector = concatenate([_fc_protein, _cnn_protein, _fc_cmpd, _cnn_cmpd], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature_1')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[ prot_vec, prot_char_in, mol_vec, cmpd_smile_in], output=outputs)
    return model

def pcm3_1(dropout_value=0.5, ecfp=1, vec=1, num_filters = 32, embed_dim=16, kernel_size=4 ):
    init = 'lecun_normal'
    CNN_act ='elu'
    active_func ='elu'

    prot_vec = Input(shape=(300,), name='prot_vec')
    mol_vec  = Input(shape=(1024*ecfp +300*vec ,), name='mol_vec')
    prot_char_in  = Input(shape=(700,), name='prot_char_in')
    cmpd_smile_in = Input(shape=(100,), name='cmpd_smile_in')

    _fc_protein = Dense(1024, activation=active_func, kernel_initializer=init, name='Prot_feature1')(prot_vec)
    _fc_protein=Dropout(0.1)(_fc_protein)
    _fc_protein = Dense(1024, activation=active_func, kernel_initializer=init, name='Prot_feature2')(_fc_protein)
    _fc_protein=Dropout(dropout_value)(_fc_protein)
    _fc_protein = Dense(512, activation=active_func, kernel_initializer=init, name='Prot_feature3')(_fc_protein)
    _fc_protein=Dropout(dropout_value)(_fc_protein)

    _fc_cmpd = Dense(2048,  activation=active_func, kernel_initializer=init, name='Cmpd_feature1')(mol_vec)
    _fc_cmpd=Dropout(0.1)(_fc_cmpd)
    _fc_cmpd = Dense(1024, activation=active_func, kernel_initializer=init, name='Cmpd_feature2')(_fc_cmpd)
    _fc_cmpd=Dropout(dropout_value)(_fc_cmpd)
    _fc_cmpd = Dense(512, activation=active_func, kernel_initializer=init, name='Cmpd_feature3')(_fc_cmpd)
    _fc_cmpd=Dropout(dropout_value)(_fc_cmpd)
############

    prot_embedd = Embedding(input_dim=len(UNIQUE_PROT), output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_PROT)(prot_char_in)
    cmpd_embedd = Embedding(input_dim=len(UNIQUE_CMPD)+1, output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_CMPD)(cmpd_smile_in)

    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=1 , kernel_initializer=init)(prot_embedd)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=2 , kernel_initializer=init)(prot_i)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=4 , kernel_initializer=init)(prot_i)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=8 , kernel_initializer=init)(prot_i)

    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=1 , kernel_initializer=init)(cmpd_embedd)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=2 , kernel_initializer=init)(cmpd_i)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=4 , kernel_initializer=init)(cmpd_i)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=8 , kernel_initializer=init)(cmpd_i)

    _cnn_protein= Flatten()(prot_i)
    _cnn_cmpd= Flatten()(cmpd_i)
    _cnn_protein= Flatten()(prot_i)
    _cnn_cmpd= Flatten()(cmpd_i)
    merged_vector = concatenate([_fc_protein, _cnn_protein, _fc_cmpd, _cnn_cmpd], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature_1')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[ prot_vec, prot_char_in, mol_vec, cmpd_smile_in], output=outputs)
    return model


def pcm4_1(dropout_value=0.5, ecfp=1, vec=1, num_filters = 32, embed_dim=16, kernel_size=4 ):
    init = 'lecun_normal'
    CNN_act ='elu'
    active_func ='elu'

    prot_vec = Input(shape=(300,), name='prot_vec')
    mol_vec  = Input(shape=(1024*ecfp +300*vec ,), name='mol_vec')
    prot_char_in  = Input(shape=(700,), name='prot_char_in')
    cmpd_smile_in = Input(shape=(100,), name='cmpd_smile_in')

    _fc_protein = Dense(1024, activation=active_func, kernel_initializer=init, name='Prot_feature1')(prot_vec)
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
############

    prot_embedd = Embedding(input_dim=len(UNIQUE_PROT), output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_PROT)(prot_char_in)
    cmpd_embedd = Embedding(input_dim=len(UNIQUE_CMPD)+1, output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_CMPD)(cmpd_smile_in)

    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=1, kernel_initializer=init)(prot_embedd)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=2, kernel_initializer=init)(prot_i)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=4, kernel_initializer=init)(prot_i)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=8, kernel_initializer=init)(prot_i)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=16, kernel_initializer=init)(prot_i)

    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=1, kernel_initializer=init)(cmpd_embedd)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=2, kernel_initializer=init)(cmpd_i)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=4, kernel_initializer=init)(cmpd_i)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=8, kernel_initializer=init)(cmpd_i)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=16, kernel_initializer=init)(cmpd_i)

    _cnn_protein= Flatten()(prot_i)
    _cnn_cmpd= Flatten()(cmpd_i)
    
    merged_vector = concatenate([_fc_protein, _cnn_protein, _fc_cmpd, _cnn_cmpd], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature_1')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[ prot_vec, prot_char_in, mol_vec, cmpd_smile_in], output=outputs)
    return model