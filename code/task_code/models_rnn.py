from keras.models import Model
from keras.layers import Dense,Input, Dropout, Embedding, Flatten
from keras.layers.merge import concatenate
from keras.layers import Flatten, Input, Dropout, LSTM, CuDNNLSTM, Bidirectional, Embedding

MAXLEN_PROT= 700
UNIQUE_PROT= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

MAXLEN_CMPD= 100
UNIQUE_CMPD = ['#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4',
       '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'F', 'G', 'H',
       'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'V', 'W', 'Z',
       '[', '\\', ']', 'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'n', 'o','p', 'r', 's', 't', 'u']


def pcm4_1(dropout_value=0.5, n_units = 64, embed_dim=16, is_Bidirect=False):
    init = 'lecun_normal'
    active_func = 'elu'
    
    prot_lstm = CuDNNLSTM(units=n_units, return_state=False)
    cmpd_lstm = CuDNNLSTM(units=n_units, return_state=False)
    if is_Bidirect:
        prot_lstm = Bidirectional(prot_lstm, merge_mode='concat')
        cmpd_lstm = Bidirectional(cmpd_lstm, merge_mode='concat')
    
    char_prot_in  = Input(shape=(700,), name='Prot')
    smile_cmpd_in = Input(shape=(100,), name='Compound')
    prot_embedd = Embedding(input_dim=len(UNIQUE_PROT), output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_PROT)(char_prot_in)
    cmpd_embedd = Embedding(input_dim=len(UNIQUE_CMPD)+1, output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_CMPD)(smile_cmpd_in)

    prot_ = prot_lstm(prot_embedd)
    cmpd_ = cmpd_lstm(cmpd_embedd)

    merged_vector = concatenate([prot_, cmpd_], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[char_prot_in, smile_cmpd_in], output=outputs)
    return model


def pcm4_1_drop(dropout_value=0.5, n_units = 32, embed_dim=16, is_Bidirect=False):
    init = 'lecun_normal'
    active_func = 'elu'
    
    prot_lstm = LSTM(units=n_units, return_state=False, recurrent_dropout=0.4)
    cmpd_lstm = LSTM(units=n_units, return_state=False, recurrent_dropout=0.4)
    if is_Bidirect:
        prot_lstm = Bidirectional(prot_lstm, merge_mode='concat')    
        cmpd_lstm = Bidirectional(cmpd_lstm, merge_mode='concat')
    
    char_prot_in  = Input(shape=(700,), name='Prot')
    smile_cmpd_in = Input(shape=(100,), name='Compound')
    prot_embedd = Embedding(input_dim=len(UNIQUE_PROT), output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_PROT)(char_prot_in)
    cmpd_embedd = Embedding(input_dim=len(UNIQUE_CMPD)+1, output_dim=embed_dim, embeddings_initializer=init, input_length=MAXLEN_CMPD)(smile_cmpd_in)

    prot_ = prot_lstm(prot_embedd)
    cmpd_ = cmpd_lstm(cmpd_embedd)

    merged_vector = concatenate([prot_, cmpd_], axis=1, name='merge')
    output = Dense(256, activation=active_func, kernel_initializer=init, name='merge_feature')(merged_vector)
    output=Dropout(dropout_value)(output)
    outputs = Dense(2, activation='softmax', name='output')(output)
    
    model = Model(input=[char_prot_in, smile_cmpd_in], output=outputs)
    return model