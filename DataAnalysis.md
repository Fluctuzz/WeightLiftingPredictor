Introduction
------------

In this simple data analysis a random forest model is trained on a
Weight Lifting Exercises Data set. The data set contains many variables
relating to motion of the person during the exercise. As a predictor
value `classe` will be used. It is a factor variable from A to E
indicating how well the exercise was performed.

Data Preparation
----------------

At first the libraries and data set get loaded into R and a parallel
computing cluster is initiated. Also data is split into a training and
test set to estimated the out of sample error after the model is
trained.

    library(data.table)
    library(dplyr)
    library(caret)
    library(parallel)
    library(doParallel)
    library(ggplot2)
    #Setting seed for Reproducibility 
    set.seed(314)

    cluster <- makeCluster(detectCores() - 1)
    registerDoParallel(cluster)

    data <-  read.csv("pml-training.csv")

    train_index <- createDataPartition(data$classe, p=0.9, list = FALSE)

    train <- data[train_index,]
    test <- data[-train_index,]

In the next section columns unrelated to the quality of the
exercise/`classe` get removed (e.g username, time stamps), because these
variables don’t give an indication about `classe`. Furthermore columns
with a lot of empty or missing values get removed for the same reason.

    uninformative_cols <- grep("^kurtosis|^skewness|^max|^min|^amplitude", names(train), value=TRUE)

    unrelated_cols <- c("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp",
                    "new_window", "num_window", "X")

    train <- train[ , !(names(train) %in% c(uninformative_cols,unrelated_cols))]
    test <- test[ , !(names(test) %in% c(uninformative_cols,unrelated_cols))]

    #Remove Columns less than 10000 non NAs
    train <- train[colSums(!is.na(train)) > 1000]
    test <- test[colSums(!is.na(train)) > 1000] #Have to use same subset as in train

Model Training
--------------

The training process of the model happens in several steps. At first
Principal Component Analysis is used on the data set. This reduces
columns from 53 to 25 and thereby speeding up the training process and
removing linear relationships between the measurements. After that a
random forest model is trained with cross validation with 5 folds. The
cross validation helps to ensure that the model isn’t overfitting to the
particulate data set.

    control <- trainControl(method="cv", number=5, allowParallel = TRUE)
    fit <- train(classe ~ ., 
                 data=train,
                 preProcess = "pca",
                 method="rf", 
                 trControl = control)

    #Stoping parallel cluster
    stopCluster(cluster)
    registerDoSEQ()

Results & Evaluation
--------------------

### Confusion matrix of fitted model

    confusionMatrix.train(fit)

    ## Cross-Validated (5 fold) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 28.3  0.3  0.0  0.0  0.0
    ##          B  0.0 18.7  0.2  0.0  0.1
    ##          C  0.0  0.3 17.0  0.7  0.1
    ##          D  0.1  0.0  0.1 15.6  0.1
    ##          E  0.0  0.0  0.0  0.0 18.1
    ##                             
    ##  Accuracy (average) : 0.9774

The accuracy of 0.98 is pretty good considering the model was only
trained on 25 variables.

### Interpretation of the model

The box plot shows that the median of the first principal component is
higher for the `classe` A.

    pca_inputs <- predict(fit$preProcess, train)
    ggplot(data=train, aes(x=classe, y=pca_inputs$PC1, color=classe)) + ylab("PC1") + geom_boxplot()

![](DataAnalysis_files/figure-markdown_strict/unnamed-chunk-4-1.png)

Furthermore a scatter plot between only the first and second principal
component shows a clustering in 5 groups. A clear distinction between
the `classe` isn’t identifiable, suggesting that the training data has
more dimensions.

    ggplot(data=pca_inputs, aes(x=PC1, y=PC2, color=train$classe))+ labs(color="classe")+ geom_point(alpha=0.4)

![](DataAnalysis_files/figure-markdown_strict/unnamed-chunk-5-1.png)

### Estimated out of sample error

The model performs about the same on the test set, suggesting the the
model doesn’t overfit on the training data.

    test_pred <- predict(fit, test)
    confusionMatrix(test_pred, test$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   A   B   C   D   E
    ##          A 552  10   0   0   0
    ##          B   2 366   5   0   0
    ##          C   4   3 333   9   4
    ##          D   0   0   3 311   3
    ##          E   0   0   1   1 353
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.977           
    ##                  95% CI : (0.9694, 0.9832)
    ##     No Information Rate : 0.2847          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.971           
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9892   0.9657   0.9737   0.9688   0.9806
    ## Specificity            0.9929   0.9956   0.9876   0.9963   0.9988
    ## Pos Pred Value         0.9822   0.9812   0.9433   0.9811   0.9944
    ## Neg Pred Value         0.9957   0.9918   0.9944   0.9939   0.9956
    ## Prevalence             0.2847   0.1934   0.1745   0.1638   0.1837
    ## Detection Rate         0.2816   0.1867   0.1699   0.1587   0.1801
    ## Detection Prevalence   0.2867   0.1903   0.1801   0.1617   0.1811
    ## Balanced Accuracy      0.9911   0.9806   0.9807   0.9826   0.9897
