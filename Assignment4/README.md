# a4

k_nearest_neighbours.py
- This class is used for implementing the K Nearest Neighbours classifier. 

Here is a description of the methods used in the program:

- fit()
1. In the fit() method, I am initializing the input data

- predict()
1. In the predict() method, I first iterate for every data point, and calculate the distance of the given input with all data points, and sort the results by distance in descending order. Next, I get the first k data points along with their class.
2. If the weighting strategy is 'uniform', all k data points are given the same weight while calculating the majority vote
3. If the weighting strategy is 'distance', each data point is weighted by the reciprocal of their distance to the input. To account for points whose distance might be zero, I add a smoothing parameter between 0 and 1. 
4. Finally, the class of the data point having the highest majority is given as the prediction.


multilayer_perceptron.py
- This class is used for implementing the MLP classifier.

- _initialize()
1. This method initializes the train features and one-hot encodes the labels vector.
2. It also randomly initializes the model with weights drawn from a uniform distribution between 0 and 1.

- fit()
1. This method first calls the initialize() method to initialize the network
2. Next, the for loop runs for a designated number of iterations, wherein each iteration first executes a forward pass of the train data through the network. After the forward pass, the loss function is used to calculate the loss, which is further used for error gradient calculation.
3. Then, the backpropagation algorithm is used to update the weights of the hidden and output layers using this error gradient. 
4. To visualize model training, the training loss at every 20th epoch is stored in a list. 

- predict()
1. Using the trained model, this method just executes a forward pass with the given data and returns a list with the predictions for the given input data.


Challenges faced:
1. Implementing backpropagation was challenging as a lot of error gradients had to be calculated whose formulas were hard to implement.
2. Keeping the value of error gradients in check was a challenge as due to the problem of exploding or vanishing gradients during backpropagation, either the gradient became too small or too large. Clipping checks had to be used to counter this problem. 
