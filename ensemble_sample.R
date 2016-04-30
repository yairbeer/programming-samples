# KAGGLE COMPETITION

# We are adding in the argument stringsAsFactors=FALSE, since we have some text fields

NewsTrain = read.csv("NYTimesBlogTrain.csv", stringsAsFactors=FALSE)
NewsTest = read.csv("NYTimesBlogTest.csv", stringsAsFactors=FALSE)

NewsTrain$NewsDesk = as.factor(NewsTrain$NewsDesk) 
NewsTrain$SectionName = as.factor(NewsTrain$SectionName) 
NewsTrain$SubsectionName = as.factor(NewsTrain$SubsectionName)

NewsTest$NewsDesk = as.factor(NewsTest$NewsDesk) 
NewsTest$SectionName = as.factor(NewsTest$SectionName) 
NewsTest$SubsectionName = as.factor(NewsTest$SubsectionName) 

# changing word count to log

NewsTrain$WordCount = log(NewsTrain$WordCount)
NewsTest$WordCount = log(NewsTest$WordCount)

NewsTrain$WordCount = sub(-Inf, 0, NewsTrain$WordCount)
NewsTest$WordCount = sub(-Inf, 0, NewsTest$WordCount)

NewsTrain$WordCount = as.numeric(NewsTrain$WordCount)
NewsTest$WordCount = as.numeric(NewsTest$WordCount)

# To convert the date/time to something R will understand, you can use the following commands:

NewsTrain$PubDate = strptime(NewsTrain$PubDate, "%Y-%m-%d %H:%M:%S")
NewsTest$PubDate = strptime(NewsTest$PubDate, "%Y-%m-%d %H:%M:%S")

# The second argument tells the strptime function how the data is formatted. 
# If you opened the file in Excel or another spreadsheet software before loading it into R, you might have to adjust the format. 
# See the help page ?strptime for more information.

# Now that R understands this field, there are many different attributes of the date and time that you can extract.
# For example, you can add a variable to your datasets called "Weekday" that contains the day of the week that the article was published (0 = Sunday, 1 = Monday, etc.), by using the following commands:

NewsTrain$Weekday = NewsTrain$PubDate$wday
NewsTest$Weekday = NewsTest$PubDate$wday

NewsTrain$Hour = NewsTrain$PubDate$hour
NewsTest$Hour = NewsTest$PubDate$hour

# working on texts
library(tm)
library(SnowballC)

# Create corpus

corpus_train_Headline = Corpus(VectorSource(NewsTrain$Headline))
corpus_test_Headline = Corpus(VectorSource(NewsTest$Headline))

corpus_train_Snip = Corpus(VectorSource(NewsTrain$Snippet))
corpus_test_Snip = Corpus(VectorSource(NewsTest$Snippet))

corpus_train_Abs = Corpus(VectorSource(NewsTrain$Abstract))
corpus_test_Abs = Corpus(VectorSource(NewsTest$Abstract))

# Convert to lower-case

corpus_train_Headline = tm_map(corpus_train_Headline, tolower)
corpus_test_Headline = tm_map(corpus_test_Headline, tolower)

corpus_train_Snip = tm_map(corpus_train_Snip, tolower)
corpus_test_Snip = tm_map(corpus_test_Snip, tolower)

corpus_train_Abs = tm_map(corpus_train_Abs, tolower)
corpus_test_Abs = tm_map(corpus_test_Abs, tolower)

# IMPORTANT NOTE: If you are using the latest version of the tm package, you will need to run the following line before continuing (it converts corpus to a Plain Text Document).
# This is a recent change having to do with the tolower function that occurred after this video was recorded.
corpus_train_Headline = tm_map(corpus_train_Headline, PlainTextDocument)
corpus_test_Headline = tm_map(corpus_test_Headline, PlainTextDocument)

corpus_train_Snip = tm_map(corpus_train_Snip, PlainTextDocument)
corpus_test_Snip = tm_map(corpus_test_Snip, PlainTextDocument)

corpus_train_Abs = tm_map(corpus_train_Abs, PlainTextDocument)
corpus_test_Abs = tm_map(corpus_test_Abs, PlainTextDocument)

# Remove punctuation
corpus_train_Headline = tm_map(corpus_train_Headline, removePunctuation)
corpus_test_Headline = tm_map(corpus_test_Headline, removePunctuation)

corpus_train_Snip = tm_map(corpus_train_Snip, removePunctuation)
corpus_test_Snip = tm_map(corpus_test_Snip, removePunctuation)

corpus_train_Abs = tm_map(corpus_train_Abs, removePunctuation)
corpus_test_Abs = tm_map(corpus_test_Abs, removePunctuation)

corpus_train_Headline = tm_map(corpus_train_Headline, removeWords, stopwords("english"))
corpus_test_Headline = tm_map(corpus_test_Headline, removeWords, stopwords("english"))

corpus_train_Snip = tm_map(corpus_train_Snip, removeWords, stopwords("english"))
corpus_test_Snip = tm_map(corpus_test_Snip, removeWords, stopwords("english"))

corpus_train_Abs = tm_map(corpus_train_Abs, removeWords, stopwords("english"))
corpus_test_Abs = tm_map(corpus_test_Abs, removeWords, stopwords("english"))

# Stem document 
corpus_train_Headline = tm_map(corpus_train_Headline, stemDocument)
corpus_test_Headline = tm_map(corpus_test_Headline, stemDocument)

corpus_train_Snip = tm_map(corpus_train_Snip, stemDocument)
corpus_test_Snip = tm_map(corpus_test_Snip, stemDocument)

corpus_train_Abs = tm_map(corpus_train_Abs, stemDocument)
corpus_test_Abs = tm_map(corpus_test_Abs, stemDocument)

# Create Matrix
dtm_train_Headline = DocumentTermMatrix(corpus_train_Headline)
dtm_test_Headline = DocumentTermMatrix(corpus_test_Headline)

dtm_train_Snip = DocumentTermMatrix(corpus_train_Snip)
dtm_test_Snip = DocumentTermMatrix(corpus_test_Snip)

dtm_train_Abs = DocumentTermMatrix(corpus_train_Abs)
dtm_test_Abs = DocumentTermMatrix(corpus_test_Abs)

dtm_train_Headline = removeSparseTerms(dtm_train_Headline, 0.98)

dtm_train_Snip = removeSparseTerms(dtm_train_Snip, 0.97)

dtm_train_Abs = removeSparseTerms(dtm_train_Abs, 0.97)

dtm_train_Headline =  as.data.frame(as.matrix(dtm_train_Headline))
dtm_test_Headline =  as.data.frame(as.matrix(dtm_test_Headline))

dtm_train_Snip =  as.data.frame(as.matrix(dtm_train_Snip))
dtm_test_Snip =  as.data.frame(as.matrix(dtm_test_Snip))

dtm_train_Abs =  as.data.frame(as.matrix(dtm_train_Abs))
dtm_test_Abs =  as.data.frame(as.matrix(dtm_test_Abs))

# Make all variable names R-friendly
colnames(dtm_train_Headline) =  make.names(colnames(dtm_train_Headline))
colnames(dtm_test_Headline) =  make.names(colnames(dtm_test_Headline))

colnames(dtm_train_Snip) =  make.names(colnames(dtm_train_Snip))
colnames(dtm_test_Snip) =  make.names(colnames(dtm_test_Snip))

colnames(dtm_train_Abs) =  make.names(colnames(dtm_train_Abs))
colnames(dtm_test_Abs) =  make.names(colnames(dtm_test_Abs))

colnames(dtm_train_Headline) = paste0("H", colnames(dtm_train_Headline))
colnames(dtm_train_Snip) = paste0("S", colnames(dtm_train_Snip))
colnames(dtm_train_Abs) = paste0("A", colnames(dtm_train_Abs))

colnames(dtm_test_Headline) = paste0("H", colnames(dtm_test_Headline))
colnames(dtm_test_Snip) = paste0("S", colnames(dtm_test_Snip))
colnames(dtm_test_Abs) = paste0("A", colnames(dtm_test_Abs))

dtmTrain = cbind(dtm_train_Headline, dtm_train_Snip, dtm_train_Abs)
dtmTest = cbind(dtm_test_Headline, dtm_test_Snip, dtm_test_Abs)

dtmTrain$Popular = NewsTrain$Popular
dtmTest$Popular = NewsTest$Popular

dtmTrain$NewsDesk = NewsTrain$NewsDesk
dtmTest$NewsDesk = NewsTest$NewsDesk

dtmTrain$SectionName = NewsTrain$SectionName
dtmTest$SectionName = NewsTest$SectionName

dtmTrain$SubsectionName = NewsTrain$SubsectionName
dtmTest$SubsectionName = NewsTest$SubsectionName

dtmTrain$WordCount = NewsTrain$WordCount
dtmTest$WordCount = NewsTest$WordCount

dtmTrain$Weekday = as.factor(NewsTrain$Weekday)
dtmTest$Weekday = as.factor(NewsTest$Weekday)

dtmTrain$Hour = as.factor(NewsTrain$Hour)
dtmTest$Hour = as.factor(NewsTest$Hour)

dtmTrain$Popular = as.factor(NewsTrain$Popular)

# Creating dummy data
newsDeskDummy = as.data.frame(model.matrix( ~ NewsDesk - 1, data=dtmTrain ))
sectionNameDummy = as.data.frame(model.matrix( ~ SectionName - 1, data=dtmTrain ))
subsectionNameDummy = as.data.frame(model.matrix( ~ SubsectionName - 1, data=dtmTrain ))

newsDeskDummyTest = as.data.frame(model.matrix( ~ NewsDesk - 1, data=dtmTest ))
sectionNameDummyTest = as.data.frame(model.matrix( ~ SectionName - 1, data=dtmTest ))
subsectionNameDummyTest = as.data.frame(model.matrix( ~ SubsectionName - 1, data=dtmTest ))

# Removing columns only in the training files
newsDeskDummy$NewsDeskNational = NULL
newsDeskDummy$NewsDeskSports = NULL
cbind(colnames(newsDeskDummy), colnames(newsDeskDummyTest))

sectionNameDummy$SectionNameSports = NULL
sectionNameDummy$SectionNameStyle = NULL
cbind(colnames(sectionNameDummy), colnames(sectionNameDummyTest))

subsectionNameDummy$"SubsectionNameFashion & Style" = NULL
subsectionNameDummy$SubsectionNamePolitics = NULL
cbind(colnames(subsectionNameDummy), colnames(subsectionNameDummyTest))

dtmTrain = cbind(dtmTrain, newsDeskDummy, sectionNameDummy, subsectionNameDummy)

dtmTrain$NewsDesk = NULL
dtmTrain$SectionName = NULL
dtmTrain$SubsectionName = NULL

names(dtmTrain) <- sub(" ", ".", names(dtmTrain))
names(dtmTrain) <- sub("/", ".", names(dtmTrain))
names(dtmTrain) <- sub(" ", ".", names(dtmTrain))

# prepare test data
library(dplyr)
# Selecting only common words
dtmTest_selected <- select(dtmTest, HX2015, Hdaili, Hday, Hfashion, Hnew, Hreport, Htoday, Hweek, Hyork, Sarticl,
                           Scan, Scompani, Sday, Sfashion, Sfirst, Sintern, Smake, Snew, Soffer, Sone,
                           Sphoto, Spresid, Sreport, Ssaid, Ssenat, Sshare, Sshow, Sstate, Stake, Stime,
                           Sweek, Swill, Syear, Syork, Aarticl, Acan, Acompani, Aday, Afashion, Afirst,
                           Aintern, Amake, Anew, Aoffer, Aone, Aphoto, Apresid, Areport, Asaid, Asenat,
                           Ashare, Ashow, Astate, Atake, Atime, Aweek, Awill, Ayear, Ayork, Hour, Weekday,
                           NewsDesk, SectionName, SubsectionName, WordCount)

newsDeskDummyTest = as.data.frame(model.matrix( ~ NewsDesk - 1, data=dtmTest ))
sectionNameDummyTest = as.data.frame(model.matrix( ~ SectionName - 1, data=dtmTest ))
subsectionNameDummyTest = as.data.frame(model.matrix( ~ SubsectionName - 1, data=dtmTest ))


dtmTest_selected = cbind(dtmTest_selected, newsDeskDummyTest, sectionNameDummyTest,
                         subsectionNameDummyTest)

dtmTest_selected$NewsDesk = NULL
dtmTest_selected$SectionName = NULL
dtmTest_selected$SubsectionName = NULL

names(dtmTest_selected) <- sub(" ", ".", names(dtmTest_selected))
names(dtmTest_selected) <- sub("/", ".", names(dtmTest_selected))
names(dtmTest_selected) <- sub(" ", ".", names(dtmTest_selected))
cbind(colnames(dtmTrain),colnames(dtmTest_selected))

library(caTools)
set.seed(144)
split = sample.split(dtmTrain$Popular, SplitRatio = 0.7)
train_train = subset(dtmTrain, split==TRUE)
train_test = subset(dtmTrain, split==FALSE)

# Random Forest
library(randomForest)

# Build optimized random forest model
set.seed(1000)
mtry.best <- tuneRF(train_train[, -60], train_train[, 60], mtryStart = 16, 
                    stepFactor = 2, ntreeTry = 500, improve = 0.01)

mtry.best = mtry.best[which.min(mtry.best[,2]),1]

set.seed(1000)
RForest = randomForest(as.factor(Popular) ~ ., data=train_train, ntree = 2000, mtry = mtry.best)

# Make predictions
PredictForest = predict(RForest, newdata = train_test, type = "prob")

# Checking AUC training set score
library(ROCR)
pred = prediction(PredictForest[,2], train_test$Popular)
as.numeric(performance(pred, "auc")@y.values)

# SVM Classifier
library(e1071)
svm_model <- svm(as.factor(Popular) ~ ., data=train_train, probability=TRUE)
svm_predictions = predict(svm_model, newdata = train_test, probability=TRUE)
svm_prob = attr(svm_predictions, "probabilities")[,1]
pred = prediction(svm_prob, train_test$Popular)
as.numeric(performance(pred, "auc")@y.values)

# Checking correlation
cor(svm_prob,PredictForest[,2])

# Checking best weights on ensemble of RF and SVC
ensambles_a = sapply(0:100, function(i){
  set.seed(1000)
  tot_pred = (i/100)*(svm_prob)+((100-i)/100)*(PredictForest[,2])
  pred = prediction(tot_pred, train_test$Popular)
  as.numeric(performance(pred, "auc")@y.values)
})

ensambles_a



# Logistic Regression after RF
c_algo_train = as.data.frame(cbind(dtmTrain, PredictForest = PredictForest_train[,2]))
c_algo_test = as.data.frame(cbind(dtmTest_selected, PredictForest = PredictForest[,2]))

# Linear regression of forward step of SVC and RF predctions
null=glm(Popular~1, data = c_algo_train, family = binomial)
full=glm(Popular~ . * WordCount * PredictForest, data = c_algo_train, family = binomial)
C_Log_reg_step = step(null, scope=list(lower=null, upper=full), direction="forward", steps = 15)
summary(C_Log_reg_step)

PredictLog_C = predict(C_Log_reg_step, newdata = c_algo_test, type = "response")
PredictLog_C_train = predict(C_Log_reg_step, type = "response")
pred = prediction(PredictLog_C_train, c_algo_train$Popular)
as.numeric(performance(pred, "auc")@y.values)

# Preparing submission, clipping x<0 and x>1 probabilities accordingly
pred_new = sapply(1:length(PredictLog_C), function(i){
  max(PredictForest[i, 2], 0)
})
pred_new = sapply(1:length(PredictLog_C), function(i){
  min(pred_new[i], 1)
})
ID = NewsTest$UniqueID


#  Write submission file for Kaggle
MySubmission = data.frame(UniqueID = ID, Probability1 = PredictLog_C)
write.csv(MySubmission, "Submission_RF_tuneRF_Log_AGP_SGP_AGnews_AGedit_HGpol.csv", row.names=FALSE)


# You should upload the submission "SubmissionSimpleLog.csv" on the Kaggle website to use this as a submission to the competition