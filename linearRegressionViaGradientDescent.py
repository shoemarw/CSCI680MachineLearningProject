# Methods for performing linear regression using gradient descent.
# Does not require numpy.
import math

#////////////////////////////////////////////////////////////////
#															   //
#			HELPER METHODS FOR VECTOR OPERATIONS			   //
#////////////////////////////////////////////////////////////////
def addv(u,v):
	"""Returns the sum of vectors u and v.
	"""
	dimension = len(u)
	if not(len(v) == dimension):
		print("You can only add vectors with the same dimension")
		return None

	sum = [0] * dimension
	for i in range(0,dimension):
		sum[i] = u[i] + v[i]

	return sum

def dot(u,v):
	"""Returns the dot product of u and v.
	"""
	dimension = len(u)
	if not(len(v) == dimension):
		print("You can only dot vectors with the same dimension")
		return None

	prod = 0
	for i in range(0, dimension):
		prod += u[i]*v[i]

	return prod

def scale(c, u):
	""" Scales the vector u by the scalar c.
	"""
	dimension = len(u)
	v = [0] * dimension
	for i in range(0, dimension):
		v[i] = c*u[i]
	return v

#////////////////////////////////////////////////////////////////
#															   //
#			METHODS FOR PERFORMING LINEAR REGRESSION		   //
#					USING GRADIENT DESCENT 					   //
#////////////////////////////////////////////////////////////////


# FUNCTIONS WITH EXPONENTIALLY DECAYING LEARNING RATES.
def stochastic_exp(alpha, beta, epochs, X, Y, T0):
	""" Implements stochasitc gradient descent for linear regression.
		Uses an exponentially decaying learning rate. The exponential
		decay is characterized by the paramater beta. beta should be
		greater than zero or it will cause divergence! 
		alpha  : learning rate
		beta   : each iteration alpha is scaled by e^(-beta)
		epochs : the number of epochs to train
		X      : feature variables. Each of the n elements of X is a
		         d dimensional vector representing a feature. 
		Y      : target variables
	"""
	m = len(X)
	d = len(X[0])
#	T = [0] * d    # initialize the weights (parameters) to 0's
	T = T0

	# Iterate the desired number of epochs.
	for l in range(0, epochs):
		# For each data point update the weights.
		for i in range(0,m):
			# Compute the update.
			update = scale(alpha*(Y[i] - dot(T,X[i])), X[i])
			# Update the weights for each data point.
			T = addv(T,update)
			# shrink the learning rate for next time.
			alpha = alpha*math.exp(-beta)

	# Return the weights		
	return T

def batch_exp(alpha, beta, epochs, X, Y, T0):
	""" Implements batch gradient descent for linear regression.
		alpha  : learning rate
		beta   : each iteration alpha is scaled by e^(-beta)
		epochs : the number of epochs to train
		X      : feature variables. Each of the n elements of X is a
		         d dimensional vector representing a feature. 
		Y      : target variables
	"""
	n = len(X)
	d = len(X[0])
#	T = [0] * d    # initialize the weights (parameters) to 0's
	T = T0

	for l in range(0, epochs):
		# Iterate through all training data to compute the 
		# update vector
		update = [0] * d
		for i in range(0,n):
			summand = scale((Y[i] - dot(T,X[i])), X[i])
			update = addv(update, summand)
		# Update the weights after all training data is considered.
		T = addv(T, scale(alpha, update))
		# Shrink the learning rate for next time.
		alpha = alpha*math.exp(-beta)

	# Return the weights
	return T

def stochastic_goodStep(alpha, beta, eps, epochs, X, Y):
	""" Implements stochasitc gradient descent for linear regression.
		Works like sochastic gradient descent with exponential decay
		rate. Only takes steps that decrease the MSE by at least eps. 
		alpha  : learning rate
		beta   : each iteration alpha is scaled by e^(-beta)
		eps.   : a threshold for how good an update must be before we
				 make an update.
		epochs : the number of epochs to train
		X      : feature variables. Each of the n elements of X is a
		         d dimensional vector representing a feature. 
		Y      : target variables
	"""
	m = len(X)
	d = len(X[0])
	T = [0] * d    # initialize the weights (parameters) to 0's

	mse = MSE(X,T,Y)

	# Iterate the desired number of epochs.
	for l in range(0, epochs):
		# For each data point update the weights.
		for i in range(0,m):
			# Compute the update.
			update = scale(alpha*(Y[i] - dot(T,X[i])), X[i])
			# Compute what the update would be.
			Tprime = addv(T,update)
			err    = MSE(X,Tprime,Y)
			# Only update if we are improving
			if (mse - err) > eps:
				# Update the weights for each data point.
				T = Tprime
				mse = err
			# shrink the learning rate for next time.
			alpha = alpha*math.exp(-beta)

	# Return the weights		
	return T

#///////////////////////////////////////////////////////////////////////
#																	  //
#						PREDICTION FUNCTION 						  //
#///////////////////////////////////////////////////////////////////////
def hypothesis(x, T):
	""" Predicts the label given weights (T) and unlabeled data in 
		the context of a regression problem.
	"""
	return dot(x, T)

#///////////////////////////////////////////////////////////////////////
#																	  //
#						     ERROR FUNCTION  						  //
#///////////////////////////////////////////////////////////////////////
def MSE(X,T,Y):
	""" Computes the mean squared error of the weights T with respect to the
    	data X and corresponding label/responses Y.
	"""
	error = 0
	num_data_points = len(X)
	for i in range(num_data_points):
		value = hypothesis(X[i],T) - Y[i]
		# Doing it this way so the result will be inf if need be 
		# (instead of throwing an exception by using value**2).
		error = error + 0.5*value*value 
	return error/num_data_points

def ONE(X,T,Y):
	""" Gives the one-normed error of the weights T with respect to the data
		X and the corresponding label/responses Y.
	"""
	error = 0
	num_data_points = len(X)
	for i in range(num_data_points):
		error  = error + abs(hypothesis(X[i],T) - Y[i])
	return error

def INE(X,T,Y):
	""" Gives the infinity-normed error of the weights T with respect to the
		data X and the corresponding label/responses Y. Also returns the 
		median of the errors.
	"""
	errors = list()
	num_data_points = len(X)
	for i in range(num_data_points):
		errors.append(abs(hypothesis(X[i],T) - Y[i]))
	errors.sort()
	return max(errors), errors[num_data_points//2]

def AverageError(X,T,Y):
	""" Gives the average error of the weights T with respect to the
		data X and the corresponding label/responses Y.
	"""
	return ONE(X,T,Y)/len(X)


#///////////////////////////////////////////////////////////////////////
#							TEST 									  //
#///////////////////////////////////////////////////////////////////////
def linearRegressionTest():
	#Use training data {(5,3), (6,5), (1,2), (8,4), (3,3)}
	X = [   [1,5], \
			[1,6], \
			[1,1], \
			[1,8], \
			[1,3]   ]
	Y = [3, 5, 2, 4, 3]

	a = float(input("Enter the desired learning rate: "))
	e = int(input("Enter the desired number of epochs: "))
	print("Learning Rate: " + str(a) + "\nEpochs: " + str(e))

	print("Using batch gradient descent: \n")
	T = batch_const(a, e, X, Y)
	for i in range(0, len(T)):
		print("T_" + str(i) + "= " + str(T[i]) + "\n")

	print("Mean squared error:", MSE(X,T,Y))

	print("Using stochastic gradient descent: \n")
	T = stochastic_const(a, e, X, Y)
	for i in range(0, len(T)):
		print("T_" + str(i) + "= " + str(T[i]) + "\n")

		print("Mean squared error:", MSE(X,T,Y))

if __name__ == '__main__':
	linearRegressionTest()
