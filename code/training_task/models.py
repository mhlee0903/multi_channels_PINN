from keras.models import Model
from keras.layers import Dense, concatenate,Input, Dropout, Embedding, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation
from keras.layers.merge import concatenate


UNIQUE_PROT= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
MAXLEN_PROT= 700
MAXLEN_CMPD= 100
UNIQUE_CMPD = ['#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4',
       '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'F', 'G', 'H',
       'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'V', 'W', 'Z',
       '[', '\\', ']', 'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'n', 'o','p', 'r', 's', 't', 'u']
INIT = 'lecun_normal'
CNN_act ='elu'
ACTIVE_func ='elu'

def get_fc(_input, _n_nodes, _dropout_value, _name):
    _output = Dense(int(_n_nodes), activation=ACTIVE_func, kernel_initializer=INIT, name=_name)(_input)
    _output = Dropout(_dropout_value)(_output)
    return _output

input_name_list = ['feat1', 'feat2', 'feat3', 'feat4']

def get_PINN(PROT_type='H', CMPD_type='H', SL_length=2, CL_length=1,
			dropout_value=0.5, ecfp=1, vec=1, num_filters = 16, embed_dim=32, 
            kernel_size=12, nb_cnn_prot=5, nb_cnn_cmpd=2, n_nodes=1024, padding='causal',
            ):


    dropout_list = [0.1, dropout_value, dropout_value, dropout_value]
    
    if SL_length == 1:
    	nodes_list = [int(n_nodes)]
    elif SL_length == 2:
    	nodes_list = [int(n_nodes), int(n_nodes//4)]
    elif SL_length == 3:
    	nodes_list = [int(n_nodes), int(n_nodes//2), int(n_nodes//4)]
    elif SL_length == 4:
    	nodes_list = [int(n_nodes), int(n_nodes//2), int(n_nodes//2), int(n_nodes//4)]

    if CL_length == 1:
    	nodes_list_CL = [int(n_nodes//4)]
    elif CL_length == 2:
    	nodes_list_CL = [int(n_nodes//4), int(n_nodes//4)]


    merge_list = []
    input_list = []
    if PROT_type == 'V' or PROT_type == 'H':
    	prot_vec = Input(shape=(300,), name='prot_vec')
    	input_list.append(prot_vec)

        for _n_nodes, _name, _dropout_value in zip(nodes_list, input_name_list, dropout_list):
        	_fc_protein = get_fc( _input=prot_vec, _n_nodes=_n_nodes, _name='Prot_'+_name, _dropout_value=_dropout_value)
        merge_list.append(_fc_protein)

    if PROT_type == 'R' or PROT_type == 'H':
        prot_char_in  = Input(shape=(700,), name='prot_char_in')
        input_list.append(prot_char_in)

        prot_embedd = Embedding(input_dim=len(UNIQUE_PROT), output_dim=embed_dim, embeddings_initializer=INIT, input_length=MAXLEN_PROT)(prot_char_in)
        prot_i = Conv1D(num_filters, kernel_size, activation=ACTIVE_func, padding=padding,dilation_rate=1, kernel_initializer=INIT, name='conv_prot_0')(prot_embedd)
        for i in range(nb_cnn_prot):
            prot_i = Conv1D(num_filters, kernel_size, activation=ACTIVE_func, padding=padding,dilation_rate=2**(i+1), kernel_initializer=INIT, name='conv_prot_{}'.format(i+1))(prot_i)
        _cnn_protein= Flatten()(prot_i)
        merge_list.append(_cnn_protein)

    if CMPD_type == 'V' or CMPD_type ==  'H':
        mol_vec  = Input(shape=(1024*ecfp +300*vec ,), name='mol_vec')
        input_list.append(mol_vec)

        for _n_nodes, _name, _dropout_value in zip(nodes_list, input_name_list, dropout_list):
        	_fc_cmpd = get_fc( _input=mol_vec, _n_nodes=_n_nodes, _name='Cmpd_'+_name, _dropout_value=_dropout_value)
        merge_list.append(_fc_cmpd)	

    if CMPD_type == 'R' or CMPD_type == 'H':
        cmpd_smile_in = Input(shape=(100,), name='cmpd_smile_in')
        input_list.append(cmpd_smile_in)

        cmpd_embedd = Embedding(input_dim=len(UNIQUE_CMPD)+1, output_dim=embed_dim, embeddings_initializer=INIT, input_length=MAXLEN_CMPD)(cmpd_smile_in)
        cmpd_i = Conv1D(num_filters, kernel_size, activation=ACTIVE_func, padding=padding,dilation_rate=1, kernel_initializer=INIT, name='conv_cmpd_0')(cmpd_embedd)
        for i in range(nb_cnn_cmpd):
            cmpd_i = Conv1D(num_filters, kernel_size, activation=ACTIVE_func, padding=padding,dilation_rate=2**(i+1), kernel_initializer=INIT, name='conv_cmpd_{}'.format(i+1))(cmpd_i)

        _cnn_cmpd= Flatten()(cmpd_i)
        merge_list.append(_cnn_cmpd)

    output = concatenate(merge_list, axis=1, name='merge')

    for _n_nodes, _name in zip(nodes_list_CL, input_name_list):
    	output = get_fc( _input=output, _n_nodes=_n_nodes, _name='output_'+_name, _dropout_value= dropout_value)
    outputs = Dense(2, activation='softmax', name='output')(output)
    model = Model(input=input_list, output=outputs)
    return model

