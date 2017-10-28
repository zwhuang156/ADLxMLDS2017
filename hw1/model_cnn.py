import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Conv2D, Masking, Dense, Dropout, Activation, BatchNormalization, Bidirectional, TimeDistributed
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers.core import Reshape


frame_size = 777


class Path_define():
    TrainSet = "fbank"
    train_data_path = "data/" + TrainSet + "/train.ark"
    train_label_path = "data/train.lab"
    test_data_path = "data/" + TrainSet + "/test.ark"
    phone_to_num_path = "data/48phone_char.map"
    model_path = "model/" + TrainSet + "_model_CNN.h5"
    ans_path = "ans/" + TrainSet + ".csv"
    sample_path = "data/sample.csv"
    map48_path = "data/48phone_char.map"
    map39_path = "data/48_39.map"


def Load_train_data(train_data_path, train_label_path, phone_to_num_path):
    # load data
    X_train = pd.read_table(train_data_path, sep=' ', header=None)
    X_train = np.array(X_train.values)
    Y_train = pd.read_table(train_label_path, sep=',', header=None)
    Y_train = np.array(Y_train.values)
    phone_to_num = pd.read_table(phone_to_num_path, sep='\t', header=None, index_col=0)
    phone_to_num = phone_to_num.to_dict(orient='dict')
    # transfer Y_train to number
    for i in range(len(Y_train)):
        Y_train[i][1] = phone_to_num[1][Y_train[i][1]]

    # sort trainning data
    X_train = sorted(X_train, key=lambda x:x[0])
    Y_train = sorted(Y_train, key=lambda x:x[0])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return (X_train, Y_train)


def Reconstruct_and_Padding_X(data):
    new_data = np.zeros((3696,frame_size,data.shape[1]-1))
    sentence = 0
    frame = 0
    for i in range(len(data)):
        now = data[i][0].split("_")
        if i==0:
            new_data[sentence,frame] = data[i][1:]
            frame += 1
        else:
            if now[0]==before[0] and now[1]==before[1]:
                new_data[sentence,frame] = data[i][1:]
                frame += 1
            if now[0]!=before[0] or now[1]!=before[1]:
                frame = 0
                sentence += 1
                new_data[sentence,frame] = data[i][1:]
                frame += 1
        before = now

    return new_data

def Reconstruct_and_Padding_Y(data):
    new_data = np.zeros((3696,frame_size,48),dtype=np.int)
    sentence = 0
    frame = 0
    for i in range(len(data)):
        now = data[i][0].split("_")
        if i==0:
            new_data[sentence,frame] = np_utils.to_categorical(data[i][1], num_classes=48)
            frame += 1
        else:
            if now[0]==before[0] and now[1]==before[1]:
                new_data[sentence,frame] = np_utils.to_categorical(data[i][1:], num_classes=48)
                frame += 1
            if now[0]!=before[0] or now[1]!=before[1]:
                frame = 0
                sentence += 1
                new_data[sentence,frame] = np_utils.to_categorical(data[i][1:], num_classes=48)
                frame += 1
        before = now

    return new_data

"""
def Reconstruct_and_Padding_X_for_conv2D(data):
    new_data = np.zeros((3696,frame_size,data.shape[1]-1,1))
    sentence = 0
    frame = 0
    for i in range(len(data)):
        now = data[i][0].split("_")
        if i==0:
            for j in range(1,len(data[i])):
                new_data[sentence,frame,j-1] = data[i][j]
            frame += 1
        else:
            if now[0]==before[0] and now[1]==before[1]:
                for j in range(1,len(data[i])):
                    new_data[sentence,frame,j-1] = data[i][j]
                frame += 1
            if now[0]!=before[0] or now[1]!=before[1]:
                frame = 0
                sentence += 1
                for j in range(1,len(data[i])):
                    new_data[sentence,frame,j-1] = data[i][j]
                frame += 1
        before = now

    return new_data
"""

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    path = Path_define()
    # Load data
    X_train, Y_train = Load_train_data(path.train_data_path, path.train_label_path, path.phone_to_num_path)

    # Reconstruct & Padding
    X_train = Reconstruct_and_Padding_X(X_train)
    Y_train = Reconstruct_and_Padding_Y(Y_train)


    # Build RNN
    time_steps = X_train.shape[1]
    feature_size = X_train.shape[2]
    filter_num = 150


    model = Sequential()

    model.add(Conv1D(
                filters = filter_num,
                kernel_size = 100,
                input_shape=(time_steps, feature_size),
                padding = "same"))              
    model.add(Masking(mask_value=0.))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(units = 512)))
    model.add(Bidirectional(LSTM(
                units = 512,
                return_sequences = True,
                activation='relu')))
    model.add(Bidirectional(LSTM(
                units = 512,
                return_sequences = True,
                activation='relu')))                 
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(
                units = 48,
                activation = 'softmax')))
    print(model.summary())

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    #Starting trainning
    print(" Start trainning~~~~ ")
    model.fit(  X_train,
                Y_train,
                batch_size=32,
                epochs=2
                )




    model.save(path.model_path)



    # handle repeat phone















