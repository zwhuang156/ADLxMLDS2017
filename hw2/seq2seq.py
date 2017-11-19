import os
import json
import re
import operator
import numpy as np
from random import shuffle

import tensorflow as tf
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Masking, Dense, Dropout, BatchNormalization, TimeDistributed, Lambda, concatenate
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
#from keras.utils import np_utils


class Path_define():
    train_data_path = "data/training_data/feat"
    train_label_path = "data/training_label.json"

"""    
def Load_train_data_old(train_data_path, train_label_path, vocab_dict, max_seq_length):
    
    X_train = []
    Y_train_in = []
    Y_train_out = []
    label = open(train_label_path, 'r')
    label = json.load(label)
    
    for file in os.listdir(train_data_path):
        filename = file.replace(".npy", "")
        
        for i in range(1450):
            if label[i]['id']==filename:
                size = len(label[i]['caption'])
                for j in range(size):
                    temp_in = to_onehot(label[i]['caption'][j], vocab_dict, max_seq_length)
                    temp_out = temp_in[:]
                    del temp_in[-1]
                    del temp_out[0]
                    Y_train_in.append(temp_in)
                    Y_train_out.append(temp_out)
        
        feature = np.load(os.path.join(train_data_path, file))
        X_train.append(feature)
        for i in range(size-1):
            X_train.append([])
            X_train[i+1] = X_train[0]
        
    X_train = np.array(X_train)
    Y_train_in = np.array(Y_train_in)
    Y_train_out = np.array(Y_train_out)

    return X_train, Y_train_in, Y_train_out    
"""    
    
def common_vocab_OneHot(train_label_path, vocab_size, reverse=0):
    vocab = {}
    file = open(train_label_path, 'r')
    label = json.load(file)
    for i in range(len(label)):
        del label[i]['id']
    label = json.dumps(label)
    label = re.split('\[|]|}|{|\.|:| |,|"', label)
    for i in range(len(label)):
        if label[i] not in vocab:
            vocab[label[i]] = 1
        else:
            vocab[label[i]] += 1
    del vocab['caption']
    del vocab['']
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse = True)
    ''' vocab became a List '''
    vocab = vocab[:vocab_size-2]
    vocab_dict = {}
    reverse_dict = {}
    init = np.zeros((vocab_size),dtype=np.int)
    for i in range(len(vocab)):
        one_hot = np.copy(init)
        one_hot[i] = 1
        vocab_dict[vocab[i][0]] = one_hot
        reverse_dict[i] = vocab[i][0]
    vocab_dict['unknown'] = init
    reverse_dict[0] = vocab[i][0]
    bos = np.copy(init)
    bos[vocab_size-2] = 1
    eos = np.copy(init)
    eos[vocab_size-1] = 1
    vocab_dict['<BOS>'] = bos
    reverse_dict[vocab_size-2] = '<BOS>'
    vocab_dict['<EOS>'] = eos
    reverse_dict[vocab_size-1] = '<EOS>'
    
    if reverse==0:
        return vocab_dict
    elif reverse==1:
        return reverse_dict

    
def to_onehot(sequence, vocab_dict, max_seq_length):
    temp = sequence[:]
    sequence = sequence.replace(".", "")
    sequence = sequence.replace(",", "")
    sequence = sequence.split(" ")

    new_seq = []
    '''
    # pre padding
    for i in range(80):
        new_seq.append(vocab_dict['unknown'])
    '''
    new_seq.append(vocab_dict['<BOS>'])
    for i in range(len(sequence)):
        if i >= max_seq_length-2:
            break
        if sequence[i] in vocab_dict:
            new_seq.append(vocab_dict[sequence[i]])
    new_seq.append(vocab_dict['<EOS>'])
    # padding to max_seq_length
    for i in range(max_seq_length-len(new_seq)):
        '''for i in range(max_seq_length-(len(new_seq)-80)):'''
        new_seq.append(vocab_dict['unknown'])
    
    
    return new_seq
    
    
    
def Load_train_data(filelist, train_data_path, train_label_path, vocab_dict, max_seq_length):
    
    X_train = []
    Y_train_in = []
    Y_train_out = []
    label = open(train_label_path, 'r')
    label = json.load(label)
    
    '''padding = np.zeros((max_seq_length-1,4096), dtype=np.int)'''
    for file in filelist:
        filename = file.replace(".npy", "")
        
        for i in range(1450):
            if label[i]['id']==filename:
                size = len(label[i]['caption'])
                for j in range(size):
                    temp_in = to_onehot(label[i]['caption'][j], vocab_dict, max_seq_length)
                    '''temp_out = temp_in[80:]'''
                    temp_out = temp_in[:]
                    temp_in = temp_in[0:1]
                    #del temp_in[-1]
                    del temp_out[0]
                    Y_train_in.append(temp_in)
                    Y_train_out.append(temp_out)
        
        feature = np.load(os.path.join(train_data_path, file))
        '''feature = np.concatenate((feature, padding), axis=0)    # padding first LSTM input to ((80+max_seq_length), 4096)'''
        for i in range(size):
            X_train.append(feature)
        
    X_train = np.array(X_train)
    Y_train_in = np.array(Y_train_in)
    Y_train_out = np.array(Y_train_out)
    
    """
    print(X_train.shape)
    print(Y_train_in.shape)
    print(Y_train_out.shape)
    """
    return X_train, Y_train_in, Y_train_out



def slice(input, type):
    if type=='encode':
        print(input.shape)
        return input[:,0:80,:]
   
    if type=='decode':  
        return input[:,80:,:]

def get_loss(mask_value):
    mask_value = K.variable(mask_value)
    def masked_categorical_crossentropy(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character '#'
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        # multiply categorical_crossentropy with the mask
        loss = K.categorical_crossentropy(y_true, y_pred) * mask

        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)
    return masked_categorical_crossentropy

        

if __name__ == '__main__':
    path = Path_define()
    max_seq_length = 35
    vocab_size = 3000
    batch_size = 145
    epochs = 5

    OneHot_table = common_vocab_OneHot(path.train_label_path, vocab_size)
    
    #X_train, Y_train_in, Y_train_out = Load_train_data(path.train_data_path, path.train_label_path, OneHot_table, max_seq_length)
    
    
    
    # define model
    """   
    encoder_inputs = Input(shape=(80, 4096))
    encoder = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)  
    encoder_states = [state_h, state_c]
    
    decoder_inputs = Input(shape=(max_seq_length-1, vocab_size))
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))
    final_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], final_outputs)
    """

    unit_1st = 256
    '''
    X_input = Input(shape=(80 + max_seq_length-1, 4096))
    lstm_out = LSTM(unit_1st, return_sequences=True)(X_input)
    Y_inputs = Input(shape=(80 + max_seq_length-1, vocab_size))
    
    concatenate_data = concatenate([lstm_out,Y_inputs], axis=-1)
    
    encoder_inputs = Lambda(slice, output_shape=(80, unit_1st+vocab_size), arguments={'type':'encode'})(concatenate_data)
    decoder_inputs = Lambda(slice, output_shape=(max_seq_length-1, unit_1st+vocab_size), arguments={'type':'decode'})(concatenate_data)
    
    encoder_outputs, state_h, state_c = LSTM(256, return_state=True)(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_outputs = LSTM(256, return_sequences=True)(decoder_inputs, initial_state=encoder_states)
    final_outputs = Dense(vocab_size, activation='softmax')(decoder_outputs)
      
    model = Model([X_input, Y_inputs], final_outputs)
    '''
    
    """
    en_input = Input(shape=(80, 4096))
    en_1_out, en_1_state_h, en_1_state_c = LSTM(unit_1st, return_sequences=True, return_state=True)(en_input)
    en_1_state = [en_1_state_h, en_1_state_c]
    en_2_out, en_2_state_h, en_2_state_c = LSTM(unit_1st, return_sequences=True, return_state=True)(en_1_out)
    en_2_state = [en_2_state_h, en_2_state_c]
    
    de_input = Input(shape=(max_seq_length-1, vocab_size))
    de_1_out = LSTM(unit_1st, return_sequences=True)(de_input, initial_state=en_1_state)
    en_2_out = LSTM(unit_1st, return_sequences=True)(de_1_out, initial_state=en_2_state)
    
    final_outputs = Dense(vocab_size, activation='softmax')(en_2_out)
    model = Model([en_input, de_input], final_outputs)
    """
    
    en_input = Input(shape=(80, 4096))
    en_1_out, en_1_state_h, en_1_state_c = LSTM(unit_1st, return_sequences=True, return_state=True, activation='relu')(en_input)
    en_1_state = [en_1_state_h, en_1_state_c]
    en_2_out, en_2_state_h, en_2_state_c = LSTM(unit_1st, return_sequences=True, return_state=True, activation='relu')(en_1_out)
    en_2_state = [en_2_state_h, en_2_state_c]

    
    state_1 = en_1_state
    state_2 = en_2_state
    all_outputs = []
    de_input = Input(shape=(1, vocab_size))
    input = de_input
    for i in range(max_seq_length-1) :
        de_1_out, de_1_state_h, de_1_state_c = LSTM(unit_1st, return_sequences=True, return_state=True, activation='relu')(de_input, initial_state=state_1)
        state_1 = [de_1_state_h, de_1_state_c]
        de_2_out, de_2_state_h, de_2_state_c = LSTM(unit_1st, return_sequences=True, return_state=True, activation='relu')(de_1_out, initial_state=state_2)
        state_2 = [de_2_state_h, de_2_state_c]
        pred_output = Dense(vocab_size, activation='softmax')(de_2_out)
        all_outputs.append(pred_output)
        input = pred_output
    
    final_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)    
    model = Model([en_input, de_input], final_outputs)
    
    print(model.summary())
    
    mask_categorical_crossentropy = get_loss(np.zeros((vocab_size), dtype=np.int))

    opt = Adam(lr=0.001)
    #opt = 'rmsprop'
    model.compile(  optimizer=opt,
                    loss = mask_categorical_crossentropy,
                    metrics=['accuracy'])


    # Starting training
    filelist = []
    count = 0
    for i in range(epochs):
        print("---------- epoch %d/%d ----------" % (i+1, epochs))
        all_file = os.listdir(path.train_data_path)
        shuffle(all_file)
        for file in all_file:
            """for file in os.listdir(path.train_data_path):"""
            filelist.append(file)
            count += 1
            if count%batch_size==0:
                X_train, Y_train_in, Y_train_out = Load_train_data(filelist, path.train_data_path, path.train_label_path, OneHot_table, max_seq_length)
                filelist = []
                model.fit([X_train, Y_train_in], Y_train_out, epochs = 1, verbose=2)
                del X_train, Y_train_in, Y_train_out
    

    model.save("model/model.h5")



    
    
    
    
    
    
    
    
    
    
    
    