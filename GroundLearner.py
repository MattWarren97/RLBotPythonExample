import sklearn
import csv

""" Data Format: 
carLocX_0, carLocY_0, carDirX_0, carDirY_0, carVelX_0, carVelY_0,
carLocX_1, carLocY_1, carDirX_1, carDirY_1, carVelX_1, carVelY_1,
throttle, steer, time
 """

def readCSV():
	fileName = "MovementData/1540728691.116972.csv"
	features = []
	targets = []
	with open(fileName, 'r') as csvFile:
		dataReader = csv.reader(csvFile, lineterminator='\r\n\n')
		next(dataReader)

		for c, row in enumerate(dataReader):
			f = []
			t = []
			if c < 15:
				print("Row: ", c, bytes(str(row), 'utf-8'))
				continue
			else:
				break
			if c==0:
				print("Header " + ", ".join(row))
				continue
			f.extend(row[0:5])
			t.extend(row[6:11])
			f.extend(row[12:14])
			print("Fetaures: ", f)
			print("Targets: ", t)
			if c % 15 == 0:
				break

			
			



def main():
	print("Hello?")
	readCSV()


if __name__ == "__main__":
    main()