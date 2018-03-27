1 Packages
==========

For our prediction analysis, we are going to use following packages.

``` r
library("caret")
library("dplyr")
library("rattle")
library("randomForest")
```

2 Data Loading
==============

After puting the data source into your work directory, run the following to load them

``` r
pml.training <- read.csv("pml-training.csv")
pml.testing <- read.csv("pml-testing.csv")
```

3 Validation data
=================

To further testing our model building, I will split 20% of the training data into validation data. Setting the seed to 1234, so we can reproduce the process

``` r
set.seed(1234)
inTrain <- createDataPartition(pml.training$classe, p=0.8)[[1]]
pml.training.sub <- pml.training[inTrain,]
pml.training.val <- pml.training[-inTrain,]
```

4 Data Cleaning
===============

After looking at the data set, I find out that quite a lot of the variables have many NAs or blank values. Because I don't want to lose many records. So I will just remove variables with many NAs or Blank values. The following is my algorihtm that will help me do that.

``` r
Na_Per <- function(x){sum(is.na(x)*1)/length(x)}
Blank_Per <- function(x){sum((x == '')*1)/length(x)}

c(Na_Per(pml.training.sub$skewness_yaw_belt), Blank_Per(pml.training.sub$skewness_yaw_belt))
```

    ## [1] 0.0000000 0.9787885

``` r
cols <- names(pml.training.sub)
col_na_blank <- data.frame(cols = cols
                           , na_percentage = rep(0,length(cols))
                           , blank_percentage = rep(0,length(cols))
                           )

for (coln in cols)
{
  exp1 <- paste("col_na_blank[col_na_blank$cols == '",coln, "', ]$na_percentage <- Na_Per(pml.training.sub$",coln,")", sep = "")
  eval(parse(text=exp1))
  exp1 <- paste("col_na_blank[col_na_blank$cols == '",coln, "', ]$blank_percentage <- Blank_Per(pml.training.sub$",coln,")", sep = "")
  eval(parse(text=exp1))
  
}

Clean_Names <- col_na_blank[col_na_blank$na_percentage == 0 & col_na_blank$blank_percentage == 0,]$cols
Clean_Names <- as.character(Clean_Names)
```

After I get all the variables with no NAs or Blanks. I then select those variables in my sub training and validation data set. Also I will need to remove all the individual identity columns and use only the variables collected from the device.

``` r
pml.training.sub.cl <- pml.training.sub[Clean_Names]
pml.training.val.cl <- pml.training.val[Clean_Names]

pml.training.sub.cl2 <- pml.training.sub.cl[c(-1,-2, -3,-4,-5,-6,-7)]
pml.training.val.cl2 <- pml.training.val.cl[c(-1,-2, -3,-4,-5,-6,-7)]
```

5 Model Fitting
===============

After gettting the data set that I want, I then come to the stage to fit my model. I am going to use two methods to fit the data. One is decision tree and the other one is random forest.

``` r
modfit_rp <- train(classe ~ ., method = "rpart", data=pml.training.sub.cl2)
modfit_rf <- randomForest(classe ~ ., data=pml.training.sub.cl2)
```

Then I will apply the models to the validation data set and see how is the model working

``` r
pre_val_rp <- predict(modfit_rp, pml.training.val.cl)
pre_val_rf <- predict(modfit_rf, pml.training.val.cl)
```

From the following we can see that the random forest model is much better than the decision tree. The random forest has accuracy equals to 0.9977 and the p value is smaller than 2.2e-16 which is significanly small.

### Decision Tree

``` r
confusionMatrix(pre_val_rp, pml.training.val.cl$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1025  310  325  290  115
    ##          B   20  271   23  102   95
    ##          C   69  178  336  251  184
    ##          D    0    0    0    0    0
    ##          E    2    0    0    0  327
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4994          
    ##                  95% CI : (0.4836, 0.5151)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3451          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9185  0.35705  0.49123   0.0000  0.45354
    ## Specificity            0.6295  0.92415  0.78944   1.0000  0.99938
    ## Pos Pred Value         0.4964  0.53033  0.33006      NaN  0.99392
    ## Neg Pred Value         0.9510  0.85698  0.88021   0.8361  0.89037
    ## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
    ## Detection Rate         0.2613  0.06908  0.08565   0.0000  0.08335
    ## Detection Prevalence   0.5264  0.13026  0.25950   0.0000  0.08386
    ## Balanced Accuracy      0.7740  0.64060  0.64033   0.5000  0.72646

### Random Forest

``` r
confusionMatrix(pre_val_rf, pml.training.val.cl$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    2    0    0    0
    ##          B    0  756    5    0    0
    ##          C    0    1  679    2    0
    ##          D    0    0    0  641    0
    ##          E    0    0    0    0  721
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9975          
    ##                  95% CI : (0.9953, 0.9988)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9968          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9960   0.9927   0.9969   1.0000
    ## Specificity            0.9993   0.9984   0.9991   1.0000   1.0000
    ## Pos Pred Value         0.9982   0.9934   0.9956   1.0000   1.0000
    ## Neg Pred Value         1.0000   0.9991   0.9985   0.9994   1.0000
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1927   0.1731   0.1634   0.1838
    ## Detection Prevalence   0.2850   0.1940   0.1738   0.1634   0.1838
    ## Balanced Accuracy      0.9996   0.9972   0.9959   0.9984   1.0000

6 Applying the Model
====================

After selecting the random forest, I will apply the model to the test data set. I will aslo do the variable clean up on the test data set. Also before predicting the test data set with the random forest model, I am going to redo the factors on the test data. Because the prediction function can only work on data set with same factors.

``` r
Clean_Names_t <- Clean_Names[Clean_Names != "classe"]

pml.testing.cl <- pml.testing[Clean_Names_t]

levels(pml.testing.cl$user_name) <- levels(pml.training.sub.cl$user_name)
levels(pml.testing.cl$cvtd_timestamp) <- levels(pml.training.sub.cl$cvtd_timestamp)
levels(pml.testing.cl$new_window) <- levels(pml.training.sub.cl$new_window)
levels(pml.testing.cl$new_window) <- levels(pml.training.sub.cl$new_window)
```

After cleaning up the data, I then run the predict funtion on the data set with random forest model. The prediction result is as following

``` r
predict(modfit_rf, pml.testing.cl)
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

7 Conclusion
============

From the quize, I know that my prediction on the testing data set is completely right. As a matter of fact, the random forest do a really good job on fitting the data set and provide an accurate prediction.
