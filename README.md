# Coding examples
This github projects consists of samples of both R and python scripting examples. The content of this script
are written by me unless explictly mentioned overwise.

## Python scripts
### classifying_ranked_sample.py
This script was used in the "Prudential Life Insurance Assessment" competition on the 11/2015 on kaggle.com
The purpose of the competition was to assess the health status of patients according to the a dataset. The
health status was a dependant ordinal variable between 1 to 8.
This script is a second tier solver and it consists of several steps:

1. Read predictions linear regression, random forest, xgboost and SVR train and test meta-estimators.
2. Stratified K-Fold cross validation: linear regression of the train meta-predictions and then to cut into ordinal classes optimally
3. Linear regression of the train meta-predictions predicting the test meta-predictions and then to cut into ordinal classes optimally.
4. Write to submission file

### computer_vision_sample.py
This script was used in the "State Farm Distracted Driver Detection" competition on the 04/2016 on kaggle.com
The purpose of the competition was to detect distracted drivers between 10 different categories for different driver.
This script consists of several steps:

1. Read images.
2. Change images from RGB to gray scale to reduce image parameters and to reduce overfitting between different drivers.
2. Conduct Monte-Carlo experiment of the training to know what is the mean and standard deviation of the
3. Plan KFold cross validation scheme by separating between different drivers in order to reduce overfitting.
4. Train Convolution Neural Network, while in each batch use the training data after minute transformations (resizing, rotating, shearing) to reduce overfitting.
5. Predict test images from the CNN and write into submission file

### heatmap_sample.py
This script is for Monte-Carlo simulations of errors in positioning estimations from multiple Access Points using maximum probability.
Where each AP have normal distributed error in measurement.
The script gives the average standard deviation and a heatmap of the standard-deviation as a function of \(X, Y\).

Technologies:

* classes
* numpy
* global and local(scipy.optimize) optimization
* Monte-Carlo Simulation
* Matplotlib

## R programming scripts
### NLP_ensemble_sample.R
This script was used in MITx's 15.071x - The Analytics Edge Spring 2015 competition.
The goal was to predict whether the article is popular or not by analyzing its headline, snippet and abstract.
This script consists of several steps:

1. Read the files
2. Parse the information from the file: Split the date into weekday + hour, convert text to lower case, remove punctuation, stem words.
3. Create a sparse matrix of word counting in each variable
4. Train/test split to validate results
5. Predict using Random Forest classifier and SVC
6. Use linear regression on the meta-predictors ensemble.
7. Write results to submission file

### model_over_time.R
This script is for simulation of errors in positioning estimation over time while the mobile station is
moving using weighted mean of crossings of couples of access points.
The script gives the plots of x\(t\), y\(t\) and the 2D track, raw estimation and filtered over time estimation.

Technologies:

* ggplot2
* time filter
* R functions

### modeling_sample.R
This script is consisted from several homework answers from week 4 of HarvardX - PH525.3x Advances Statistics for the Life Sciences on edx.org.

Technologies:

* Binomial Probability distribution
* Normal and Poisson approximation for Binomial distribution
* Maximum Likelihood Estimate