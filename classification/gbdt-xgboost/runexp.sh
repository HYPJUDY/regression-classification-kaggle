#!/bin/bash
# training and output the models
nohup /home/xgboost/xgboost classification.conf >> models_etad005_ga0_mcw9_dep12/nohup.out 2>&1 &
# Continue from Existing Model
nohup /home/xgboost/xgboost classification.conf model_in=/home/classification/models_etad01_ga0_mcw7_dep12/0900.model num_round=100 model_out=1000.model >> models_etad01_ga0_mcw7_dep12/nohup.out 2>&1 &
# output prediction task=pred 
/home/xgboost/xgboost classification.conf task=pred model_in=/home/classification/models_etad01_ga0_mcw7_dep12/0800.model
