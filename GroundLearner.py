from sklearn.neural_network import MLPRegressor
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
	#return row[6:15] #for now, ignore the ball, otherwise 0:15
	f = row[6:15]
	f.extend(row[21:30])
	return f

def getTargets(row):
	#return row[21:30] #ignore the ball, otherwise 15:30
	return row[30:32]

def getInstructions(row):
	return row[30:32] #30: throttle, 31: steer

def getTime(row):
	return row[32]

def readCSV():
	fileName = "MovementData/1540992906.2537074.csv"
	features = []
	targets = []
	prevInstr = []
	instrTime = 0
	instrTargets = []
	instrFeatures = []
	with open(fileName, 'r') as csvFile:
		print("Reading inputs from ", fileName)
		dataReader = csv.reader(csvFile, lineterminator='j')
		dataFormat = next(dataReader)

		for c, row in enumerate(dataReader):
			row = [float(i) for i in row]

			if not prevInstr: #if first element
				prevInstr = getInstructions(row)
			else:
				instructions = getInstructions(row)
				if identicalLists(instructions, prevInstr): #we don't have a new instruction
					instrTime += getTime(row)
					instrTargets = getTargets(row) #these could be the last 'targets' of the instruction
				else: #We have a new instruction, hence the end (final result) of the previous instruction (for its total duration) is known.
					#print("Index: ", c, ", new instructions: ", instructions, ". With time: ", instrTime)

					if instrFeatures:
						#first instruction is ignored, otherwise add features, targets to lists.
						#instrFeatures.append(instrTime)
						features.append(instrFeatures)
						instrTargets.append(instrTime)
						targets.append(instrTargets)
					prevInstr = instructions
					instrFeatures = getFeatures(row) #features of the new row are the start of the next instruction
					instrTime = 0


			#if c%1000 ==0:
			#	print("C: ", c)


		print(dataFormat)
		#for i, f in enumerate(features):
			#print("Feature ", i, ": ", f, "\nMapsTo: Target: ", targets[i], "\n\n")
		print("Length  of features, targets is: ", len(features), " - ", len(targets))
		return features, targets

def trainMLPRegressor(features, targets):
	mapping = list(zip(features, targets))
	random.shuffle(mapping)
	print("Have collected all Feature and Targets from data source, now to run the MLPRegressor:\n\n")

	print(features)
	print("\n\n\n\n\n\n\n\n\n\n\n")

	f_train, f_test, t_train, t_test = train_test_split(features, targets)

	print("Size of training set is: f, t: ", len(f_train), ", ", len(t_train))
	print("Size of testing set is: f, t: ", len(f_test), ", ", len(t_test))
	dataScaler = StandardScaler()
	dataScaler.fit(f_train)

	f_train = dataScaler.transform(f_train)
	f_test = dataScaler.transform(f_test)



	mlp = MLPRegressor(hidden_layer_sizes=(30, 30, 30), max_iter=10000)
	#print("From ", groundPositionPredictor.n_iter_, " iterations")
	#print("Predicting: ", groundPositionPredictor.n_outputs_, " outputs")

	mlp.fit(f_train, t_train)

	predictions = mlp.predict(f_test)
	for i, pred in enumerate(predictions):
		print("Prediction: ", pred, ", real value: ", t_test[i])

	print("TRAINING")
	trainingPredictions = mlp.predict(f_train)
	for i, pred in enumerate(trainingPredictions):
		print("Prediction: ", pred, ", real value: ", t_train[i])

	print("Training score: ", mlp.score(f_train, t_train))
	print("Testing score: ", mlp.score(f_test, t_test))




def main():
	(features, targets) = readCSV()
	trainMLPRegressor(features, targets)


if __name__ == "__main__":
    main()