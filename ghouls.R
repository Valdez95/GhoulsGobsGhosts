
library(caret)
library(tidyverse)

train <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/train.csv.zip")
train$train = TRUE
test <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/test.csv.zip")
test$train = FALSE
sample <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/sample_submission.csv.zip")
df <- bind_rows(train, test)
#train$type <- factor(train$type)


onehot <- data.frame(predict(dummyVars(" ~ type ", data=train), newdata=train))
train <- cbind(train, onehot)


## Multi-Layer Perceptron (one layer and two hidden layers)
nnet <- train(form=type~bone_length + rotting_flesh + hair_length + has_soul + color,
                 data = train,
                 method = "mlp",
                 size = 2)

preds <- predict(nnet, newdata=test)
submission <- data.frame(id=test %>% pull(id), type=preds)
write.csv(x=submission, row.names=FALSE, file="~/Desktop/mlp2layers.csv")

## Multilayer Perceptrion (5 layers with four hidden layers each except for the last layer which only has one hidden layer)
ml_nnet <- train(form=type ~ bone_length + rotting_flesh + hair_length + has_soul + color,
                 data = train,
                 method = "mlpML",
                 layer1 = 4,
                 layer2 = 4,
                 layer3 = 4,
                 layer4 = 4,
                 layer5 = 1)

preds <- predict(ml_nnet, newdata=df, type="prob")
names(preds)[1] <- "prGhost_mlp"
names(preds)[2] <- "prGhoul_mlp"
names(preds)[3] <- "prGoblin_mlp"
submission <- data.frame(id=df %>% pull(id), preds)
write.csv(x=submission, row.names=FALSE, file="~/Desktop/multilayerperceptron.csv")


## Stacked Model

nn <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/stacked_model/nn_Probs_65acc.csv")
rf <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/stacked_model/classification_submission_rf.csv")
names(rf)[1] <- "id"
mlp <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/stacked_model/multilayerperceptron.csv")
gbm <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/stacked_model/probs_gbm.csv")
names(gbm)[1] <- "id"
knn <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/stacked_model/Probs_KNN.csv")
names(knn)[1]<- "id"
xgb <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/stacked_model/xgbTree_probs.csv")
names(xgb)[1] <- "id"
svm <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/stacked_model/probs_svm.csv")
names(svm)[1] <- "id"
lreg <- vroom::vroom("~/Desktop/Kaggle/GhoulsGobsGhosts/stacked_model/LogRegPreds.csv")

new <- merge(df, rf, by="id") %>% merge(nn, by="id") %>%
  merge(mlp, by="id") %>%
  merge(gbm, by="id") %>%
  merge(knn, by="id") %>%
  merge(xgb, by="id") %>%
  merge(svm, by="id") %>%
  merge(lreg, by="id")

pp <- preProcess(x=new, method="pca")

all_ghouls <- predict(pp, new)
all_ghouls$id <- new$id


### XGBoost Tree Model ###
tr = trainControl(method="repeatedcv", number=5, repeats=3, search="grid", verbose=TRUE)
xgbTreeGrid <- expand.grid(nrounds = c(200,250,300),
                       max_depth = c(80,100,120),
                       eta = .3,
                       gamma = 2,
                       colsample_bytree = c(.4,.5),
                       min_child_weight = c(4,5,6),
                       subsample = 1) 

# using columns only included in the original dataset
ghouls.model <- train(form=type~.,
                             data = all_ghouls %>% filter(train==TRUE),
                             method = "xgbTree",
                             metric = "Accuracy",
                             tuneGrid = xgbTreeGrid,
                             trControl=tr)

preds <- predict(ghouls.model, newdata=all_ghouls %>% filter(train==FALSE))

submission <- data.frame(id=all_ghouls %>% filter(train==FALSE) %>% pull(id), type=preds)
write.csv(x=submission, row.names=FALSE, file="~/Desktop/ghouls_stacked.6.csv")

## Best Model Parameters
# nround=200
# maxdepth=100
# eta=.3
# gamma=2
# colsamp=0.5
# minchild=4


