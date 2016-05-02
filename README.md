### Coding examples
This github projects consists of samples of both R and python scripting examples. The content of this script
are written by me unless explictly written overwise.
## Python scripts
# classifying_ranked_sample.py
This script was used in the "Prudential Life Insurance Assessment" competition on the 11/2015 on kaggle.com
The purpose of the competition was to assess the health status of patients according to the a dataset. The
health status was a dependant ordinal parameter between 1 to 8.
This script was a second tier solver and it was consisted of several steps:
1. Read predictions linear regression, random forest, xgboost and SVR train and test meta-estimators.
2. Cross validate: linear regression of the train meta-predictions and then to cut into ordinal classes optimally
3. linear regression of the train meta-predictions predicting the test meta-predictions and then to cut into ordinal classes optimally.
# heatmap_sample.py
## R programming scripts
# ensemble_sample.R
# modeling_sample.R