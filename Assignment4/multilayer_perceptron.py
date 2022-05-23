# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [Ritwik Budhiraja] -- [rbudhira]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
	"""
	A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

	Attributes:
		n_hidden
			An integer representing the number of neurons in the one hidden layer of the neural network.

		hidden_activation
			A string representing the activation function of the hidden layer. The possible options are
			{'identity', 'sigmoid', 'tanh', 'relu'}.

		n_iterations
			An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

		learning_rate
			A float representing the learning rate used when updating neural network weights during gradient descent.

		_output_activation
			An attribute representing the activation function of the output layer. This is set to the softmax function
			defined in utils.py.

		_loss_function
			An attribute representing the loss function used to compute the loss for each iteration. This is set to the
			cross_entropy function defined in utils.py.

		_loss_history
			A Python list of floats representing the history of the loss function for every 20 iterations that the
			algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
			index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
			complete, the _loss_history list should have length n_iterations / 20.

		_X
			A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
			is set in the _initialize(X, y) method.

		_y
			A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
			input data used when fitting the model.

		_h_weights
			A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
			features and the hidden layer neurons.

		_h_bias
			A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
			and the hidden layer neurons.

		_o_weights
			A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
			neurons and the output layer neurons.

		_o_bias
			A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
			neuron and the output layer neurons.

	Methods:
		_initialize(X, y)
			Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
			initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

		fit(X, y)
			Fits the model to the provided data matrix X and targets y.

		predict(X)
			Predicts class target values for the given test data matrix X using the fitted classifier model.
	"""

	def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
		# Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
		activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

		# Check if the provided arguments are valid
		if not isinstance(n_hidden, int) \
				or hidden_activation not in activation_functions \
				or not isinstance(n_iterations, int) \
				or not isinstance(learning_rate, float):
			raise ValueError('The provided class parameter arguments are not recognized.')

		# Define and setup the attributes for the MultilayerPerceptron model object
		self.n_hidden = n_hidden
		self.hidden_activation = activation_functions[hidden_activation]
		self.n_iterations = n_iterations
		self.learning_rate = learning_rate
		self._output_activation = softmax
		self._loss_function = cross_entropy
		self._loss_history = []
		self._X = None
		self._y = None
		self._h_weights = None
		self._h_bias = None
		self._o_weights = None
		self._o_bias = None

	def _initialize(self, X, y):
		"""
		Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
		initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

		Args:
			X: A numpy array of shape (n_samples, n_features) representing the input data.
			y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

		Returns:
			None.
		"""

		self._X = X
		self._y = one_hot_encoding(y)

		np.random.seed(42)
		self._h_weights = np.random.rand(self._X.shape[1],self.n_hidden)    # initialising hidden weights from a uniform distribution
		self._h_bias = np.zeros((1, self.n_hidden))    # initialising hidden layer bias with zeros
		self._o_weights = np.random.rand(self.n_hidden, self._y.shape[1])    # initialising output layer weights from a uniform distribution
		self._o_bias = np.zeros((1, self._y.shape[1]))    # initialising output layer bias with zeros

	def fit(self, X, y):
		"""
		Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
		iterations.

		Args:
			X: A numpy array of shape (n_samples, n_features) representing the input data.
			y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

		Returns:
			None.
		"""

		self._initialize(X, y)

		for iter in range(self.n_iterations):
			z_h = np.dot(self._X,self._h_weights)    # z_h = W_h.X + b_h
			a_h = self.hidden_activation(z_h)    # a_h = activation(z_h)

			z_o = np.dot(a_h,self._o_weights) + self._o_bias    # z_o = W_o.z_h + b_o
			a_o = self._output_activation(z_o)    # a_o = activation(z_o)

			loss = self._loss_function(self._y,a_o)    # cross-entropy loss calculation
			if iter % 20 == 0:
				self._loss_history.append(loss)

			d_loss = a_o - self._y    # derivative of loss function w.r.t predictions
			d_a_o = self._output_activation(z_o,derivative=True)    # derivative of output activation w.r.t z_o
			d_a_h = self.hidden_activation(z_h,derivative=True)    # derivative of hidden activation w.r.t z_h

			d_lwo = np.dot(a_h.T,d_loss)    # derivative of loss w.r.t output weights
			d_lwh = np.dot(self._X.T,np.multiply((np.dot(self._o_weights,d_loss.T)).T,d_a_h))    # derivative of loss w.r.t hidden weights
			
			self._o_weights = self._o_weights - (self.learning_rate * d_lwo)    # backprop update
			self._h_weights = self._h_weights - (self.learning_rate * d_lwh)    # backprop update		

	def predict(self, X):
		"""
		Predicts class target values for the given test data matrix X using the fitted classifier model.

		Args:
			X: A numpy array of shape (n_samples, n_features) representing the test data.

		Returns:
			A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
		"""

		preds=[]
		for i in range(X.shape[0]):
			# forward pass for prediction
			x_i=np.expand_dims(X[i],axis=0)
			z_h = np.dot(x_i,self._h_weights)
			a_h = self.hidden_activation(z_h)
			z_o = np.dot(a_h,self._o_weights)
			a_o = self._output_activation(z_o)
			preds.append(np.argmax(a_o))    # getting class with maximum probability

		return np.array(preds)

