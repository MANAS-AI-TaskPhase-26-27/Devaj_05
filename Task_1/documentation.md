# Task 1.1

## Linear Regression

### Short Notes

Parameters - Weight (w) and Basis (b)
function- fw,b(X)=w1X1 + w2x2 + w3x3+ w4x4 + ..... + wnXn +b
where n is the nth column of features
Hyperparameters- Stuff u change; learning rate
Local Minima- A point where the loss is less that nearby values
GLobal Minima- The smallest loss value possible.
saddle points- where the gradient is zero but the loss increases in some directions and decreases in others.


Cost Function:
summation(f(X) - y)/ 2m
where f(x) is predicted value
      y is actual value
      m is number of x values.

Shape of the graph- Bowl Shaped in a 3d Plane where the ideal basis and weight forms the minimum height of the bowl. Can also be represented by a countour graph,  basically a birds eye view.

Gradient Descent:
Basically the model starts with an arbitary w and b and then depending on the loss, subtracts a small amount to parameters to get a new function. Based on the new function checks loss agains, corrects it and goes on for multiple iterations

algorithm; w = w - a dj(w,b)/dw and b= b - a dj(w,b)/db
where a is learning rate (by how much the model should add or subtract the small amount)
      j(w,b) is the cost function

* Should run both alg same time with same w,b value.
* as the model approachs local minimum the derivative keeps getting smaller

derivative w= d summation((fw,b(X)-y)^2)/2m/dw = d(wx+b -y)^2 / dw = summation((f(X)-y)x)/m
derivative b= d summation((fw,b(X)-y)^2)/2m/dw = d(wx+b -y)^2 / dw = summation((f(X)-y))/m

Full Batch - The model runs through the loss of every single training examples adding them up at the end of the epoch to add a small amount to the parameters to get a new line. Its more stable but computational heavy.

Stochastic- The model adds/ subtracts small amount based on loss from each training example each iteration. basically edits the relationship n times an epoch for n training example.


Normal Equation: For linear Regression without iterations.

Feature sclaing is imp as otherwise it cause the gradient descent to bounce around. Also small value x needs larger w and vice versa which is not nice. so same scale much easier to work the numbers.
methods:
1. mean normalization/ min max normalization - (x - avg value)/ (max-min)
2. z score normalization - (x - avg value)/standard deviation

ideally get it -1<x<1 or -n<x<n

### Notes related to code

Data Preprocess:
1. First convert the date column to give common date format to all the features in the column and sort the data.
2. Drop the date column as it is a weak feature to use for training the model. We sorted it first cuz we interpolate the missing data in the column later on. then for accurate results we need the data sorted in time.
3. interpolate missing values as its a time based dataset
4. conver the dataframe to arrays
5. normalize the data using zscore normalization.

Functions:
1. compute loss function to find MSE. It is better than MAE as it gets smaller as it converges allowing smooth convergence..
2. computer derivative fucntion to find the derivative. the derivative of mean of ssummation of (prediction - y)^2 is mean of summation of (prediction -y).x
3. gradient descent function runs gradient descent for n number of times. the learning rate was taken as 0.1 as its not too high and not too long allowing effective convergence ( found this value after some trial and error). Same is the case for number of iterations.

* We started of with w as an array of 0's and basis as 0
* for derivative with took transpose of the error to multiply with x as it is less resource intensive and much faster than using a loop to individually multiply dot product of each row and add them up at the end.

Plotting graphs:
plt.plot()- plots a line graph
np.linalg.norm() - finds the norm of a matrix. used to find the size of the matrix. smaller it is the better is the convergence


* the basis and weight gradient are very small showing the graph has almost converged and further iterations is not going to move the parameters much

* in test i have decided to do the r2 value to check on how accurate the model is. it follows the equation: 1- [(y - prediction)^2/ (y-mean(y))^2]


# Task 1.2

## Logistic Regression

### Short Notes

Parameters - Weight (w) and Basis (b)
function- fw,b(X)=w1X1 + w2x2 + w3x3+ w4x4 + ..... + wnXn +b
where n is the nth column of features

but we cant solve the graph by a straight line as it involves binary classification with either 1 or 0. so the values can only be [0,1].
To get values between 0 and 1 we use the sigmoid function hence logisitic regression

for the prediction to be 0 to 1, the equation used in fx_b is not enough. so the hypothesis h(w,b)= 1/ [1- e^(-fx_b)]. this results in a prediction value between 0 and 1.



Cost Function for maximum log likelihood:
summation[(y*logh(w,b)) + (1-y)(1-log h(w,b))]/m
where f(x) is predicted value
      y is actual value
      m is number of x values.


Shape of the graph- Bowl Shaped in a 3d Plane where the ideal basis and weight forms the minimum height of the bowl. Can also be represented by a countour graph,  basically a birds eye view.

One - Hot encode:
Used to convert categorical data to numbers so that the machine understands it. It basically assigns a column for each label and gives 1 when the feature coincides. 

Gradient Ascent:
Basically the model starts with an arbitary w and b and then depending on the loss, adds a small amount to parameters to get a new function. Based on the new function checks loss agains, corrects it and goes on for multiple iterations. 
* Note- Gradient descent also works

algorithm; w = w + a dj(w,b)/dw and b= b + a dj(w,b)/db
where a is learning rate (by how much the model should add or subtract the small amount)

* Should run both alg same time with same w,b value.
* as the model approachs local minimum the derivative keeps getting smaller

derivative w= d summation((h(X)-y)^2)/2m/dw = summation(y- f(X)).x/m
derivative b= d summation((h(X)-y)^2)/2m/dw = summation(f(X)-y)/m

### Notes related to code

Data preprocess:
1. Keep rows where age and police department is a perfect whole number. as some rows have police department or age with garbage values and editting them would be essentially training the model on wrong data.
2. drop weak columns
3. Fill missing values in weapons as unknown. As missing value in weapon who usually mean the weapon used was not found or identified 
4. Map the values of the closed column for better understanding by the machine. Where yes is 1 and no is 0
5. One-hot encode categorical features
6. Align test set with train set to ensure that model does not crash when features in train are not present in test set.
7. Normalize ONLY the continuous numerical columns that are not encoded.

Function:
1. computer loss by the eqn - np.sum(y * np.log(hyp) + (1 - y) * np.log(1 - hyp)) /m. np.clip() is used to prevent overflow or produce infinity. to prevent this we clip the values to produce a safe range.
2. compute derivative finds the derivative of the loss.
3. gradient ascent function for gradient ascent.

Test Metrics:
1. Accuracy: out of all the predictions, how much i got right.
2. Recall: Out of all positive how many did the model find
3. Precision: out of all the positive prediction how many are actually positive
4. Mcfadden r^2: Its the r^2 score but for classification. It indicates how good the the model compared to a baseline model that has no prediction parameters.

* When testing the predictions are converted to 1 or 0 depending on the prediction. If prediction>0.5 its 1 or else 0.
