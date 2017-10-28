import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, Conv2D, MaxPooling2D, Flatten, Masking, Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model
import csv
import sys




def Load_test_data(test_data_path, sample_path):
    # load data
    X_test = pd.read_table(test_data_path, sep=' ', header=None)
    X_test = np.array(X_test.values)
    
    ans_table = pd.read_csv(sample_path, sep=',', header=None)
    ans_table = ans_table.values
                
    return (X_test, ans_table)

def count_sentence(data):
    sentence = 1
    for i in range(len(data)):
        now = data[i][0].split("_")
        if i==0:
            pass
        else:
            if now[0]!=before[0] or now[1]!=before[1]:
                sentence += 1
        before = now   
    return sentence
   
def Reconstruct_and_Padding_X(data, sentence_num):
    abandon_table = []
    new_data = np.zeros((sentence_num, 777, data.shape[1]-1))    
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
                abandon_table.append(int(before[2]))
                sentence += 1
                frame = 0
                new_data[sentence,frame] = data[i][1:]
                frame += 1
        before = now
    abandon_table.append(int(now[2]))
    return new_data, abandon_table




if __name__=='__main__':
    # Path define
    test_data_path = sys.argv[1] + "/fbank/test.ark"
    sample_path = "sample.csv"
    model_path = "./model/fbank_model_CNN.h5"
    map48_path = sys.argv[1] + "/48phone_char.map"
    map39_path = sys.argv[1] + "/phones/48_39.map"
    ans_path = sys.argv[2]
    
    # Load test data
    X_test, final_ans = Load_test_data(test_data_path, sample_path)
    sentence_num = count_sentence(X_test)
    # Load model
    model = load_model(model_path)

    # Reconstruct & Padding
    X_test, abandon_table = Reconstruct_and_Padding_X(X_test, sentence_num)

    
    result = model.predict(X_test)
    

    sentence_lst = []
    for i in range(result.shape[0]):
        sentence_lst.append([])
        for j in range(result.shape[1]):
            sentence_lst[i].append(np.argmax(result[i,j]))
    
    
    # Handle abandon table
    result_mask = []
    for i in range(len(sentence_lst)):
        result_mask.append(sentence_lst[i][0:abandon_table[i]])
    
    
       

    # Create mapping dictionary
    ph48_index_1_to_0, ph48_index_0_to_2 = {}, {}
    map1 = open(map48_path, 'r')
    for line in map1:
        line = line.rstrip('\n')
        tmp = line.split('\t')
        ph48_index_0_to_2[tmp[0]] = tmp[2]
        ph48_index_1_to_0[tmp[1]] = tmp[0]
    map1.close()
    
    map48_39 = {}
    map2 = open(map39_path, 'r')
    for line in map2:
        line = line.rstrip('\n')
        tmp = line.split('\t')
        map48_39[tmp[0]] = tmp[1]
    map2.close()

    # map to 39 phone
    for i in range(len(result_mask)):
        for j in range(len(result_mask[i])):
            result_mask[i][j] = ph48_index_0_to_2[map48_39[ph48_index_1_to_0[str(result_mask[i][j])]]]
 
    
    # Handle redundent phone
    ans = []
    for i in range(len(result_mask)):
        ans.append([])
        for j in range(len(result_mask[i])):
            if j < len(result_mask[i])-1 and result_mask[i][j]!=result_mask[i][j+1]:
                ans[i].append(result_mask[i][j])
        ans[i].append(result_mask[i][-1])
    del result_mask
    
    # Handle silence at head and tail
    for i in range(len(ans)):
        while ans[i][0]=='L':
            del ans[i][0]
        while ans[i][-1]=='L':
            del ans[i][-1]
    # Write to final answer table        
    for i in range(len(ans)):
        row = ""
        for j in range(len(ans[i])):
            row += ans[i][j]
        final_ans[i+1][1] = row
        
        
    
 
    # Output answer
    output = open(ans_path, "w+")
    s = csv.writer(output,delimiter=',',lineterminator='\n')
    for i in range(len(final_ans)):
        s.writerow(final_ans[i]) 
    output.close()
    
    print("Answer was written !!")
    
    
    
    
    
    
    
    
    
    