#!/bin/bash
wget https://www.csie.ntu.edu.tw/~r05922156/model_y_begin.h5
python3 predict.py $1 $2 $3