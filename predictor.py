# Code for running experiments on the data. To run an experiment simply go to 
# if __name__ == '__main__': and change the values of EPOCHS, ALPHA, or BETA.
# epochs is the number of training epochs,
# alpha is the learning rate,
# and beta is the decay rate.

import csv
from linearRegressionViaGradientDescent import \
     hypothesis, MSE, ONE, INE, AverageError, stochastic_exp, batch_exp, \
     MSE, stochastic_goodStep
import matplotlib.pyplot as plt

def readData():
	""" Reads in training data from train.csv and
		returns a list of X vectors and a list of
		the target variables Y.
	""" 
	X_1 = list()
	X_2 = list()
	X_6 = list()
	X_8 = list()
	X_9 = list()
	X = list()
	Y = list()
	with open('train.csv', mode='r') as csv_file:
		csv_reader = csv.DictReader(csv_file)
		for row in csv_reader:
			# The 1 is for the bias term.
			# lot area data (feature 1)  
			X_1.append(float(row["LotArea"]))
			# year built data (feature 2)
			X_2.append(float(row["YearBuilt"]))
			# 'Gr' Living area (feature 6)
			X_6.append(float(row["GrLivArea"]))
			# OverallQual
			X_8.append(float(row["OverallQual"]))
			# OverallCond
			X_9.append(float(row["OverallCond"]))
			Y.append(float(row["SalePrice"]))

		X_plot = [X_1,X_2,X_6,X_8,X_9]

		# Normalize the data in each list
		min_1   = min(X_1)
		range_1 = max(X_1) - min_1
		min_2   = min(X_2)
		range_2 = max(X_2) - min_2
		min_6   = min(X_6)
		range_6 = max(X_6) - min_6
		min_8   = min(X_8)
		range_8 = max(X_8) - min_8
		min_9   = min(X_9)
		range_9 = max(X_9) - min_9
		# Shift everything down by the features minimum value then scale down
		# by the range so that the values are now in [0,1].
		for i in range(len(X_1)):
			X_1[i] = (X_1[i] - min_1)/range_1
			X_2[i] = (X_2[i] - min_2)/range_2
			X_6[i] = (X_6[i] - min_6)/range_6
			X_8[i] = (X_8[i] - min_8)/range_8
			X_9[i] = (X_9[i] - min_9)/range_9

		#Put the normalized data into input vectors.
		for i in range(len(X_1)):
			# The 1 is for the bias term.
			X.append([1, X_1[i],\
				 X_2[i], \
				 X_6[i], \
				 X_8[i], \
				 X_9[i] ])

	# Return the list of input vectors, the plotting data and the labels/responses
	return X, X_plot, Y


def train(X, Y, train_method, epochs, alpha, beta, T0):
	""" Trains using the inputs (X), and responses (Y), using the training method indicated
		by train_method ('s' -> stochastic or 'b' -> batch) for the number of epochs
		indicated by epochs. The learning rate is alpha every epoch by default. To use a
		learning rate which decays exponentially each epoch (alpha = alpha*e^-beta) pass
		a positive value for beta.
	"""
	if train_method == 's':
		return stochastic_exp(alpha, beta, epochs, X, Y, T0)
	elif train_method == 'b':
		return batch_exp(alpha, beta, epochs, X, Y, T0)
	else:
		print("You must provide a training method ('s' -> stochastic or 'b' -> batch)")


def printResults(X,T,Y):
	""" Prints various error measures for how well the hyperplane given by T
		fits the data given by X and Y.
	"""
	for i in range(len(T)):
		print ("T[" + str(i) + "]= " + str(T[i]))
	# Print the various error measures
	#print("Mean squared error:", MSE(X,T,Y))
	#print("One normed error:", ONE(X,T,Y))
	largest, median = INE(X,T,Y)
	#print("Infinity normed error:", largest)
	print("Median Error: ", median)
	print("Average error:", AverageError(X,T,Y))

def showPlots(X_plot, Y, T):
	""" Plots the predictions (only in the direction of each feature).
	"""
	plt.title("Only Lot Area")
	plt.xlabel("LotArea")
	plt.ylabel("PriceSold")
	plt.scatter(X_plot[0],Y)
	# Plot the prediction in this dimention, ignoring the terms associated
	# with the other features. Take into account the bias term though.
	plt.plot([0,1],   [T[0] + T[1]*0,     T[0] + T[1]*1], '-r')
	plt.show()

	plt.title("Only Year Built")
	plt.xlabel("YearBuilt")
	plt.ylabel("PriceSold")
	plt.scatter(X_plot[1],Y)
	# Plot the prediction in this dimention, ignoring the terms associated
	# with the other features. Take into account the bias term though.
	plt.plot([0,1], [T[0] + T[2]*0 , T[0] + T[2]*1], '-r')
	plt.show()

	plt.title("Only GrLivArea")
	plt.xlabel("GrLivArea")
	plt.ylabel("PriceSold")
	plt.scatter(X_plot[2],Y)
	# Plot the prediction in this dimention, ignoring the terms associated
	# with the other features. Take into account the bias term though.
	plt.plot([0,1], [T[0] + T[3]*0 , T[0] + T[3]*1], '-r')
	plt.show()

	plt.title("Only OverallQual")
	plt.xlabel("OverallQual")
	plt.ylabel("PriceSold")
	plt.scatter(X_plot[3],Y)
	# Plot the prediction in this dimention, ignoring the terms associated
	# with the other features. Take into account the bias term though.
	plt.plot([0,1], [T[0] + T[4]*0 , T[0] + T[4]*1], '-r')
	plt.show()

	plt.title("Only OverallCond")
	plt.xlabel("OverallCond")
	plt.ylabel("PriceSold")
	plt.scatter(X_plot[4],Y)
	# Plot the prediction in this dimention, ignoring the terms associated
	# with the other features. Take into account the bias term though.
	plt.plot([0,1], [T[0] + T[5]*0 , T[0] + T[5]*1], '-r')
	plt.show()


def runDecayExperiments(file_name, EPOCHS, ALPHA, BETA, T0):
	""" Runs several experiments with decaying learning rates. Produces two output files, the
		first is a csv file for the results of batch gradient descent experiments and the
		second is for the results of stochasit gradient descent.
	"""
	# Run Batch Gradient experiments.
	print("Conducting experiments with decaying learning rates.")
	X, X_plot, Y = readData()
	name = file_name + "batch_Decay.csv"
	with open(name, mode='w') as result_file:
		result_writer = csv.writer(result_file, delimiter=',')
		# Write the column names to the file
		result_writer.writerow(['epochs', 'l-rate', 'decayRate', 'biasWeight', \
			"lotAreaWeight", "YearBuiltWeight", "GrLivAreaWeight", "OverallQualWeight", \
			"OverallCondWeight", "medianError", "averageError", "largestError, MSError"])

		# Train for various combinations of epochs and learning rates BATCH.
		for epochs in EPOCHS:
			for alpha in ALPHA:
				for beta in BETA:
					print("Running new experiment...")
					print("epochs =",epochs)
					print("learning rate =", alpha)
					print("decay rate =", beta)
					T = train(X,Y,'b',epochs,alpha,beta, T0)
					# Get the experimental results ready to write to file
					largest, median = INE(X,T,Y)
					average = AverageError(X,T,Y)
					# Write the results to the file
					result_writer.writerow([epochs, alpha, beta, T[0], T[1], T[2], T[3], \
						T[4], T[5], median, average, largest, MSE(X,T,Y)])

	name = file_name + "stochastic_Decay.csv"
	with open(name, mode='w') as result_file:
		result_writer = csv.writer(result_file, delimiter=',')
		# Write the column names to the file
		result_writer.writerow(['epochs', 'l-rate', 'decayRate', 'biasWeight', \
			"lotAreaWeight", "YearBuiltWeight", "GrLivAreaWeight", "OverallQualWeight", \
			"OverallCondWeight", "medianError", "averageError", "largestError", "MSError"])

		# Train for various combinations of epochs and learning rates BATCH.
		for epochs in EPOCHS:
			for alpha in ALPHA:
				for beta in BETA:
					print("Running new experiment...")
					print("epochs =",epochs)
					print("learning rate =", alpha)
					print("decay rate =", beta)
					T = train(X,Y,'s',epochs,alpha,beta,T0)
					# Get the experimental results ready to write to file
					largest, median = INE(X,T,Y)
					average = AverageError(X,T,Y)
					# Write the results to the file
					result_writer.writerow([epochs, alpha, beta, T[0], T[1], T[2], T[3], \
						T[4], T[5], median, average, largest, MSE(X,T,Y)])

def runGoodStepExperiments(file_name, EPOCHS, ALPHA, BETA, EPS):
	""" Runs several experiments where updates are only taken if they yield desirable
	    improvement. Only uses stochastic gradient descent.
	"""
	print("Conducting experiments with 'good steps'.")
	X, X_plot, Y = readData()
	name = file_name + "goodSteps.csv"
	with open(name, mode='w') as result_file:
		result_writer = csv.writer(result_file, delimiter=',')
		# Write the column names to the file
		result_writer.writerow(['epochs', 'l-rate', 'decayRate', 'biasWeight', \
			"lotAreaWeight", "YearBuiltWeight", "GrLivAreaWeight", "OverallQualWeight", \
			"OverallCondWeight", "medianError", "averageError", "largestError", "MSError"])

		# Train for various combinations of epochs and learning rates BATCH.
		for eps in EPS:
			for epochs in EPOCHS:
				for alpha in ALPHA:
					for beta in BETA:
						print("Running new experiment...")
						print("epochs =",epochs)
						print("learning rate =", alpha)
						print("decay rate =", beta)
						T = stochastic_goodStep(alpha, beta, eps, epochs, X, Y)
						# Get the experimental results ready to write to file
						largest, median = INE(X,T,Y)
						average = AverageError(X,T,Y)
						# Write the results to the file
						result_writer.writerow([epochs, alpha, beta, T[0], T[1], T[2], T[3], \
							T[4], T[5], median, average, largest, MSE(X,T,Y)])


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#		SPECIFY THE HYPERPARAMETERS HERE         \\
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# THIS IS THE ONLY CODE THAT NEEDS TO BE CHANGED TO RUN EXPERIMENTS.

if __name__ == '__main__':
	# EPOCHS is a list containing the different numbers of epochs to use.
	EPOCHS = [40,50,70,100]
	# ALPHA is a list containing the different learning rates to use.
	ALPHA = [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
	# BETA is a list containing the different decay rates to use.
	BETA = [0, 0.0000000000001, 0.000000000001, 0.00000000001]
	EPS = [100]
	T0 = [-45805.49309287171, 41856.24378959525, 83544.70880603795, 196498.55429318393, \
			192372.80250710284, 1286.6527069494596]

	file_name = 'optStart002'

	# runConstantExperiments(file_name, EPOCHS, ALPHA)
	runDecayExperiments(file_name, EPOCHS, ALPHA, BETA, T0)
	# runGoodStepExperiments(file_name, EPOCHS, ALPHA, BETA, EPS)








