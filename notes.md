Machine Learning Notes

What is Artificial Intelligence?
Artificial Intelligence is when machine can do tasks that requires human intelligence.

Machine Learning
It’s teaching a model or a machine to predict something using data.

Label- The correct output 
Features- Data points that helps the model make a relationship

What’s the difference between supervised and unsupervised learning?
In supervised learning, the model is fed the correct output (label) along with the input data. It basically is used for the label to accurately find the relationship between features and labels. While unsupervised learning, the model is only fed features and it has to find a pattern between the features to make prediction which are then labelled.

Types of supervised learning:
1. Regression: The model finds a relationship between the features and label to predict a numerical value. 

2. Classification: The model finds a relationship to categorize the data into various classes or groups.

Types of unsupervised learning:
1. Clustering: The model predicts patterns out of the input data points and categorizes them into groups/clusters based on similarity

Semi-supervised learning:
In this type of learning, the machine is fed small dataset with label and a large dataset with unlabeled data/

Training- The model is initially trained on a large dataset to design the most accurate relationship to find the correct output.

Testing- After the model is trained it’s tested on unlabelled dataset to test the relationship.

What is a model?
A model is a collection of numbers that defines the relationship between input features and output label.

Basic Training procedure:
First data is collected, then the dataset is processed and prepared for the model to work on. The model works on patterns from the data points and makes a prediction. The predicted value is then compared with the actual value to calculate the loss. Based on the loss the model edits its parameters often the weights to reduce the loss. This process of trial and error goes on for a while and finally the model has managed to get the most accurate values to predict the relationship between the data points.
	
Preprocessing of Data:
TO check for missing values, outliers, duplicate values or any garbage values. 

Purpose of cleaning and preprocessing of Data:
IT is done mainly so that the 
(a) model doesn’t work on broken files, incorrect labels, garbage values, corrupt images, etc to make wrong predictions
(B) compute isn’t wasted on duplicate data points
(C) data types of things like dates, unit, etc are standardized 
(D) prevents sensitive information from being used for training.
(E) prevents missing information from the datasets
(F) prevents noise ( data points with no credibility) from being used for training

Imputation- replacing missing data with an estimate rather than deleting entire rows or columns. ( especially important if most of ur rows has some sort of missing data in which case deleting rows or columns can basically delete most of ur dataset)

Types of Inundation:
1. Mean Inundation- basically fills all missing values with the mean of the values of the column. Good fix but can be affected by the outliers 
2. Median Inundation- Replaces the missing values with median. Safer for datasets with outliers.
3.Backwards fill (fill)- Basically fills the missing values with the next values beneath it.
4. Forwards fill- fills the missing values with the values right above it.

Overfitting vs Underfitting:
Overfitting is when the model memorizes the training data too well and performs poorly on the test. It basically doesn’t find a lot of meaningfull patterns but instead learns the training example so well. cause by too less training data, overcomplicated model, too much training time, too much noise, or when there’s too many features and parameters.
Underfitting is when the model is too weak and can’t find any meaningful patterns. Its usually caused due to lack of important features, too much noise, very less training iterations or when the data provided doesn’t solve the problem well.

Good fit- The model can make meaningfull patterns out of the dataset and performs equally well in both training and testing.

Scaling:
In this, we transform data to fit a specific range /scale like 0-10, etc. so that the features are comparable. Eg- different currency have different value in real life, but where a certain value of the different currency given will be taken as equal by the model until we scale it so that the currency represents in real life value.
Feature Scaling- To make the values in different columns the same in size. ITs important to prevent the model from treating a particular feature superior to the other. 
For eg, 2 features age and salary where salary obviously have a bigger value than the age. This can cause the model to consider it to be a superior value. To prevent this we feature scale so that both features have same size retaining their original meaning.

Min-max scaling - Basically each value x in a dataset will be scaled to (x-xmin)/(xmin+xmax)

Normalisation:
To represent the data points as a normal distribution. Helps give all the features equal power.

Normal distribution- A bell curve where mean and median are same, most values are close to the average and half the values are above and other half is below the mean.
Box-Cow transformation- Basically convert skewed to normal distribution by a formula where a f(x)= (y^lamda -1)/lamda where lambda is not = 0 and ln y when =0. Where lamda value is chosen to find useful transformation. We use scipy module

Character Encoding-rules of converting binary numbers to human readable form (Decoding) or vice versa (encoding).
Categorical Encoding-Converting words or categories into numbers so that a model can understand them.
Eg, we have categories of colors but these categories such as red, blue , green, yellow, etc. cannot be understood by a model instead we have to convert them into numbers 
types;:
1. Label encoding- to assign each categories with a number. Preferable if the categories have an order, like big small medium. Unrelated categories like colors assigned with values can confuse the model into categorizing them with precedence. Like if red=0, yellow=1, green=2, it can confuse green>yellow>red.

Parsing of dates- Basically converting date in machine understandable format.

Evaluation Metrics:

Classification metrics:
1. Accuracy- The ratio of all the correct predictions made by the model to all the predictions made by the model. Basically how accurate the predictions made by the model is.
2. Precision- The ration of all the true positive predictions to the total positive predictions. Useful when false positives are extremely dangerous. Used to check out of all positives how many are true.
3. Recall- Used to check of all the positive predictions how many the model found. Useful where false negatives are in high stakes and costly.
4. F1 Score- Usefull if you want a balance between recall and precision. It’s the twice the ratio of product of recall and precision and their sums. It can obscure how high the recall and precision is.

Regression metrics:
1. Mean absolute error- Mean of the difference between the predicted and actual value
2. Mean Squared error- Mean of the difference between the square of predicted and actual value
3. Root Mean Squared error- Root of mean squared error




Key terms:
1. Loss- the difference between the predicted value and the actual value
2. Hyperparameters- The values we can change while training.
3. Parameters- The values changed by the model while training
4. Outliers- Data points that are irregular compared to the rest
5. Skew data- Data which are not evenly balanced around the average with usually the bulk on one side.
—————————————————==—————||———————————————————————

Linear Regression:
This type of regression, the model plots the data points on a graph and basically tries top draw the most optimal line that goes through as many data points as possible with closest proximity to the outliers.

The model usually finds the relationship by the straight line equation:
y=wx +b

Where y is the output, w is the weight or the slope of the line, x is the input value from the feature and b is a bias or the y intercept.

Out of this the model keeps adjusting the weight and bias until it gets the most accurate relationship.

Gradient Descent- A method used by the model to find the most appropriate weight and bias. IN this method the model takes an arbitrary weight and bias to plot a line and based on the loss calculated it adds or subtracts a small amount to the weight and bias to get a new line. It will take the loss once again and this keeps happening until the weight and bias gives the most appropriate line.






