import sklearn
import csv

""" Data Format: 
carLocX_0, carLocY_0, carDirX_0, carDirY_0, carVelX_0, carVelY_0,
carLocX_1, carLocY_1, carDirX_1, carDirY_1, carVelX_1, carVelY_1,
throttle, steer, time
 """

def readCSV():
	fileName = "MovementData/1540944895.429397.csv"
	features = []
	targets = []
	with open(fileName, 'r') as csvFile:
		dataReader = csv.reader(csvFile, lineterminator='j')
		dataFormat = next(dataReader)

		for c, row in enumerate(dataReader):
			f = []
			t = []
			f.extend(row[0:15])
			t.extend(row[15:30])
			f.extend(row[30:33])
			#print("Fetaures: ", f)
			#print("Targets: ", t)
			if c%1000 ==0:
				print("C: ", c)
				print("Features: ", f)
				print("Targets: ", t, "\n\n")
				#print("Row: ", row, "\n\n\n")

		print(dataFormat)

			
			



def main():
	print("Hello?")
	readCSV()


if __name__ == "__main__":
    main()