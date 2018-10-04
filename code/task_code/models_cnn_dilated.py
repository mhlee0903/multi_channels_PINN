from keras.models import Model
from keras.layers import Dense, concatenate,Input, AlphaDropout, Dropout, Embedding, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization, Activation
from keras.layers.merge import concatenate


MAXLEN_PROT= 700
UNIQUE_PROT= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

MAXLEN_CMPD= 100
UNIQUE_CMPD = ['#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4',
       '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'F', 'G', 'H',
       'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'V', 'W', 'Z',
       '[', '\\', ']', 'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'n', 'o','p', 'r', 's', 't', 'u']


def pcm3_1(dropout_value=0.5, num_filters=64, embed_dim=32, kernel_size=4):
    init = 'lecun_normal'
    active_func = 'elu'

    char_prot_in  = Input(shape=(700,), name='Prot')
    smile_cmpd_in = Input(shape=(100,), name='Compound')
    prot_embedd = Embedding(input_dim=len(UNIQUE_PROT), output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_PROT)(char_prot_in)
    cmpd_embedd = Embedding(input_dim=len(UNIQUE_CMPD)+1, output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_CMPD)(smile_cmpd_in)

    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=1 , kernel_initializer=init)(prot_embedd)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=2 , kernel_initializer=init)(prot_i)
    prot_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=4 , kernel_initializer=init)(prot_i)

    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=1 , kernel_initializer=init)(cmpd_embedd)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=2 , kernel_initializer=init)(cmpd_i)
    cmpd_i = Conv1D(num_filters, kernel_size, activation=active_func, padding='causal',dilation_rate=4 , kernel_initializer=init)(cmpd_i)

    _cnn_protein= Flatten()(prot_i)
    _cnn_cmpd= Flatten()(cmpd_i)

    merged_vector = concatenate([_cnn_protein, _cnn_cmpd], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature_1')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[char_prot_in, smile_cmpd_in], output=outputs)
    return model

def pcm4_1(dropout_value=0.5, num_filters=64, embed_dim=32, kernel_size=4):
    init = 'lecun_normal'
    active_func = 'elu'

    char_prot_in  = Input(shape=(700,), name='Prot')
    smile_cmpd_in = Input(shape=(100,), name='Compound')
    prot_embedd = Embedding(input_dim=len(UNIQUE_PROT), output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_PROT)(char_prot_in)
    cmpd_embedd = Embedding(input_dim=len(UNIQUE_CMPD)+1, output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_CMPD)(smile_cmpd_in)

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

    merged_vector = concatenate([_cnn_protein, _cnn_cmpd], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature_1')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[char_prot_in, smile_cmpd_in], output=outputs)
    return model

def pcm5_1(dropout_value=0.5, num_filters=64, embed_dim=32, kernel_size=4):
    init = 'lecun_normal'
    active_func = 'elu'

    char_prot_in  = Input(shape=(700,), name='Prot')
    smile_cmpd_in = Input(shape=(100,), name='Compound')
    prot_embedd = Embedding(input_dim=len(UNIQUE_PROT), output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_PROT)(char_prot_in)
    cmpd_embedd = Embedding(input_dim=len(UNIQUE_CMPD)+1, output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_CMPD)(smile_cmpd_in)

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

    merged_vector = concatenate([_cnn_protein, _cnn_cmpd], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature_1')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[char_prot_in, smile_cmpd_in], output=outputs)
    return model

def pcm6_1(dropout_value=0.5, num_filters=64, embed_dim=32, kernel_size=4):
    init = 'lecun_normal'
    active_func = 'elu'

    char_prot_in  = Input(shape=(700,), name='Prot')
    smile_cmpd_in = Input(shape=(100,), name='Compound')
    prot_embedd = Embedding(input_dim=len(UNIQUE_PROT), output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_PROT)(char_prot_in)
    cmpd_embedd = Embedding(input_dim=len(UNIQUE_CMPD)+1, output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_CMPD)(smile_cmpd_in)

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

    merged_vector = concatenate([_cnn_protein, _cnn_cmpd], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature_1')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[char_prot_in, smile_cmpd_in], output=outputs)
    return model