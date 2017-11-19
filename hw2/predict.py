import os
import sys
import numpy as np
import pandas as pd
import csv

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.models import load_model
from seq2seq import common_vocab_OneHot

#import keras.losses

def Load_testing_data(test_data_path, vocab_size):
    bos = np.zeros((vocab_size),dtype=np.int)
    bos[vocab_size-2] = 1
    X_test = []
    Y = []
    filename = []
    for file in os.listdir(test_data_path):
        feature = np.load(os.path.join(test_data_path, file))
        X_test.append(feature)
        Y.append(bos)
        filename.append(file.replace(".npy", ""))
 
    X_test = np.array(X_test)
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0], 1 ,vocab_size)
    print(X_test.shape)

    return X_test, Y, filename

    
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

    model_path = "model_y_begin.h5"
    test_data_path = os.path.join(sys.argv[1], "testing_data/feat")
    train_label_path = os.path.join(sys.argv[1], "training_label.json")
    ans_path = sys.argv[2]
    vocab_size = 3000
    reverse_table = common_vocab_OneHot(train_label_path, vocab_size, reverse=1)
    
    mask_categorical_crossentropy = get_loss(np.zeros((vocab_size)))
    
    """
    from keras.utils.generic_utils import get_custom_objects
    loss = get_loss(np.zeros((vocab_size)))
    get_custom_objects().update({"masked_categorical_crossentropy": loss})
    """
    
    X_test, Y, filename = Load_testing_data(test_data_path, vocab_size)
    model = load_model(model_path, custom_objects={'masked_categorical_crossentropy': mask_categorical_crossentropy})
    result = model.predict([X_test, Y])
    
    print(result.shape)
    
    ans = []
    for i in range(result.shape[0]):
        sentence = ""
        for j in range(result.shape[1]):
            if reverse_table[np.argmax(result[i][j])] != "<EOS>":
                sentence += (reverse_table[np.argmax(result[i][j])] + " ")
        ans.append(sentence)
    
    
    
    
    
    # Output answer
    output = open(ans_path, "w+")
    s = csv.writer(output,delimiter=',',lineterminator='\n')
    for i in range(len(ans)):
        s.writerow([filename[i], ans[i]]) 
    output.close()
    
    print("Answer was written !!")
    
    
