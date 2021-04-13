#Ghouls Goblins and Ghosts... Boo!

### Purpose
Predict the type of each creature (Ghoul, Goblin, or Ghost) given its bone length measurements, severity of rot, extent of soullessness, etc. 

### Files
ghouls.R contains the code for training and testing the model as well as any feature engineering on the data. 

The stacked_model directory contains submissions from various models that I used to form a stacked model. 

### Feature Engineering
For my stacked model I used the principle components from eight different model predictions as predictive variables.

### Methods
Originally I trained a multi-layer perceptron, afterwords I performed principle components analysis on the predictions from eight other model predictions. I used the predicted components as feature engineered variables and trained a boosted tree using the new data points. 

