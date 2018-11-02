from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv
import random

""" Data Format: 
carLocX_0, carLocY_0, carDirX_0, carDirY_0, carVelX_0, carVelY_0,
carLocX_1, carLocY_1, carDirX_1, carDirY_1, carVelX_1, carVelY_1,
throttle, steer, time
 """

def identicalLists(A, B):
	for i, a in enumerate(A):
		if a != B[i]:
			return False
	return True

def getFeatures(row):
	#ignore the ball; carPrevState is 6:15, carAfterState is 21:30
	f = row[6:15]
	f.extend(row[21:30])
	return f

def getTargets(row):
	#instructions are 30:32, time is 32.
	return row[30:33]

def getInstructions(row):
	return row[30:32] #30: throttle, 31: steer

def getTime(row):
	return row[32]

def readCSV():
	#fileName = "MovementData/1541104299.059675.csv" #example file (700 lines only) with uniform instructions
	fileName = "MovementData/1541114893.1472585.csv" #example file (7000 records) with uniform^2 instructions
	features = []
	targets = []
	with open(fileName, 'r') as csvFile:
		print("Reading inputs from ", fileName)
		dataReader = csv.reader(csvFile, lineterminator='j')
		dataFormat = next(dataReader)

		for c, row in enumerate(dataReader):
			row = [float(i) for i in row]

			t = getTargets(row)
			f = getFeatures(row)
			if (f[2] < 20 and f[2] > 14 and f[11] < 20 and f[11] > 14):
				#print("only ground driving")
				features.append(f)
				targets.append(t)


			#if c%1000 ==0:
			#	print("C: ", c)


		print(dataFormat)
		#for i, f in enumerate(features):
		#	print("Feature ", i, ": ", f, "\nMapsTo: Target: ", targets[i], "\n\n")
		print("Length  of features, targets is: ", len(features), " - ", len(targets))
		return features, targets

def trainMLPRegressor(features, targets):
	mapping = list(zip(features, targets))
	random.shuffle(mapping)
	print("Running MLP Regression\n\n")

	#print(features)
	#print("\n\n\n\n\n\n\n\n\n\n\n")

	f_train, f_test, t_train, t_test = train_test_split(features, targets)

	#for i,a in enumerate(t_train):
	#	if i % 100 ==0:
	#		print("Split target: ", a)

	print("Size of training set is: f, t: ", len(f_train), ", ", len(t_train))
	print("Size of testing set is: f, t: ", len(f_test), ", ", len(t_test))
	dataScaler = StandardScaler()
	dataScaler.fit(f_train)

	f_train = dataScaler.transform(f_train)
	f_test = dataScaler.transform(f_test)


	mlp = MLPRegressor(early_stopping=True, hidden_layer_sizes=(20, 20), max_iter=10000)

	mlp.fit(f_train, t_train)

	print("Training score: ", mlp.score(f_train, t_train))
	print("Testing score: ", mlp.score(f_test, t_test))

def trainLinearRegressor(features, targets):
	mapping = list(zip(features, targets))
	random.shuffle(mapping)
	print("Running Linear Regression\n\n")

	#print(features)
	#print("\n\n\n\n\n\n\n\n\n\n\n")

	f_train, f_test, t_train, t_test = train_test_split(features, targets)

	print("Size of training set is: f, t: ", len(f_train), ", ", len(t_train))
	print("Size of testing set is: f, t: ", len(f_test), ", ", len(t_test))
	dataScaler = StandardScaler()
	dataScaler.fit(f_train)

	f_train = dataScaler.transform(f_train)
	f_test = dataScaler.transform(f_test)

	lr = LinearRegression()
	lr.fit(f_train, t_train)

	#predictions = lr.predict(f_test)
	print("Training score: ", lr.score(f_train, t_train))
	print("Testing score: ", lr.score(f_test, t_test))





def main():
	(features, targets) = readCSV()
	trainLinearRegressor(features, targets)
	trainMLPRegressor(features, targets)


if __name__ == "__main__":
    main()