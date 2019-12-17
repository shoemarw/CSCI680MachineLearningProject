# Reads in csv files where the first few colunms are epochs	l-rate	
#decayRate	biasWeight lotAreaWeight YearBuiltWeight GrLivAreaWeight
#OverallQualWeight	OverallCondWeight	medianError	averageError largestError MSError
import csv
from linearRegressionViaGradientDescent import \
     hypothesis, MSE, ONE, INE, AverageError, MSE, addv, scale
from predictor import readData, printResults

def aggregateParameters(FILES, medianCutOff):
	""" Gets all the parameters contained in each file in the list FILES and averages
    	them. Only the parameters that yield a median error no larger than medianCutOff
    	are used.
	"""
	T = [0,0,0,0,0,0]
	# Keep track of how many valid sets of parameters we find. A set of parameters
	# is considered valid if they yield a median error no larger than medianCutOff.
	validParameters = 0
	# Go through every file.
	for file in FILES:
		# Open the file for reading.
		with open(file, mode='r') as csv_file:
			csv_reader = csv.DictReader(csv_file)
			# Iterate over the rows.
			for row in csv_reader:
				# Only use the weights in the row if the median error is
				# small enough.
				if float(row["medianError"]) < medianCutOff:
					# put the weights in a vector.
					weights = [float(row["biasWeight"]), \
							   float(row["lotAreaWeight"]), \
							   float(row["YearBuiltWeight"]), \
							   float(row["GrLivAreaWeight"]), \
							   float(row["OverallQualWeight"]), \
							   float(row["OverallCondWeight"])    ]
					# Add the weight to the current running total
					T = addv(T,weights)
					# Increment the number of valid parameters.
					validParameters = validParameters + 1
	# Return the aggretated vector of weights. We must divide the vector by
	# the number of valid parameter vectors we used.
	return scale(1/validParameters, T)
	

def computeErrors(T):
	""" Computes the error metrics for the parameters given by T. Uses
		the readData() method from predictor.py. The results are printed
		to the console.
	"""
	X, X_plot, Y = readData()
	printResults(X,T,Y)

if __name__ == '__main__':
	FILES = list()
	res = "resutlts0"
	sto = "stochastic_Decay.csv"
	bat = "batch_Decay.csv"
	for i in range(10, 17):
		st_str = res + str(i) + sto
		bt_str = res + str(i) + bat
		FILES.append(st_str)
		FILES.append(bt_str)
	for i in range(18, 23):
		st_str = res + str(i) + sto
		bt_str = res + str(i) + bat
		FILES.append(st_str)
		FILES.append(bt_str)

	M = [19150.0 , 19100.0, 19050.0]
	for medianCutOff in M: 
		print("\nmedianCutOff =", medianCutOff)
		T = aggregateParameters(FILES, medianCutOff)
		computeErrors(T)
		print("\n")

