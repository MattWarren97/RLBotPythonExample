import sklearn
import csv

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


def readCSV():
	fileName = "MovementData/1540944895.429397.csv"
	features = []
	targets = []
	prevInstr = []
	instrTime = 0
	instrTargets = []
	instrFeatures = []
	with open(fileName, 'r') as csvFile:
		dataReader = csv.reader(csvFile, lineterminator='j')
		dataFormat = next(dataReader)

		for c, row in enumerate(dataReader):
			row = [float(i) for i in row]
			if not prevInstr:
				prevInstr = row[30:32] #elements 30 and 31 are throttle and steer
			else:
				instructions = row[30:32]
				if identicalLists(instructions, prevInstr):
					instrTime += row[32]
					instrTargets = row[21:30] 
				else:
					#if we have reached teh end of the instruction
					if instrFeatures:
						#first instruction is ignored, otherwise add features, targets to lists.
						instrFeatures.append(instrTime)
						features.append(instrFeatures)
						targets.append(instrTargets)
						instrFeatures = row[6:15]
						instrFeatures.extend(instructions)
						prevInstr = instructions
					prevInstr = instructions
					instrFeatures = row[6:15]
					instrFeatures.extend(instructions)
					instrTime = 0


			if c%1000 ==0:
				print("C: ", c)


		print(dataFormat)
		for i, f in enumerate(features):
			print("Feature ", i, ": ", f, "\nMapsTo: Target: ", targets[i], "\n\n")
		print("Length  of feautres, targets is: ", len(features), " - ", len(targets))


			
			



def main():
	print("Hello?")
	readCSV()


if __name__ == "__main__":
    main()