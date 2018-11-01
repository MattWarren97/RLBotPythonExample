import math
import time
import csv
import random
import sys

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


def setInstrLength():
    return random.uniform(0.5, 3)

class GroundBot(BaseAgent):

    def initialize_agent(self):
        #This runs once before the bot starts up
        self.controllerState = SimpleControllerState()
        self.prevInstr = self.controllerState
        self.prevPacketSysTime = 0
        self.deltaTime = 0;
        self.initialTime = 0
        self.ticksPerInstr = 0
        self.newInstr = True
        self.instrLength = setInstrLength()
        self.dataTracker = DataTracker()

    #work out length of time since previous frame.
    def processTime(self):
        if (self.initialTime == 0):
            self.prevPacketSysTime = time.clock()
            self.initialTime = self.prevPacketSysTime
            self.instrStartTime = self.initialTime
            self.deltaTime = 0
            return False
        else:
            self.ticksPerInstr += 1
            newTime = time.clock()
            self.deltaTime = newTime-self.prevPacketSysTime
            self.prevPacketSysTime=newTime

            if newTime-self.instrStartTime >= self.instrLength:
                #time for next instruction to be given
                print(str(self.ticksPerInstr) + " physics ticks for the last instruction - Length: ", self.instrLength)
                self.ticksPerInstr = 0
                self.newInstr = True
                self.instrStartTime = newTime
                self.instrLength = setInstrLength()
            return True

    def processState(self, deltaTime, prevContrState, gameState):
        controls = [prevContrState.throttle, prevContrState.steer]
        self.dataTracker.processState(deltaTime, controls, gameState)

    def setInstructions(self, gameState):
        if self.newInstr:
            self.newInstr = False
            self.controllerState.throttle = (random.random()*2)-1 #between -1 and 1
            self.controllerState.steer = (random.random()*2)-1
            print("New instructions are throttle: " + str(self.controllerState.throttle) + ", steer: " + str(self.controllerState.steer))

        #else:
        #use same instructions
        #do nothing to self.controllerState


    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cont = self.processTime()
        if not cont:
            return self.controllerState
        gameActive = packet.game_info.is_round_active
        if not gameActive:
            return self.controllerState

        ball_location = Vector3(packet.game_ball.physics.location.x, packet.game_ball.physics.location.y, packet.game_ball.physics.location.z)
        ball_velocity = Vector3(packet.game_ball.physics.velocity.x, packet.game_ball.physics.velocity.y, packet.game_ball.physics.velocity.z)
        my_car = packet.game_cars[self.index]
        car_location = Vector3(my_car.physics.location.x, my_car.physics.location.y, my_car.physics.location.z)
        car_orientation = Vector3(my_car.physics.rotation.pitch, my_car.physics.rotation.yaw, my_car.physics.rotation.roll)
        car_velocity = Vector3(my_car.physics.velocity.x, my_car.physics.velocity.y, my_car.physics.velocity.z)
        

        currentState = GameState(ball_location, ball_velocity, car_location, car_orientation, car_velocity)

        self.processState(self.deltaTime, self.prevInstr, currentState)
        
        self.setInstructions(currentState)
        

        """car_to_ball = ball_location - car_location
        steer_correction_radians = car_direction.correction_to(car_to_ball)

        if steer_correction_radians > 0:
            # Positive radians in the unit circle is a turn to the left.
            turn = -1.0  # Negative value for a turn to the left.
        else:
            turn = 1.0

        self.controllerState.throttle = 1.0
        self.controllerState.steer = turn"""

        self.prevInstr = self.controllerState

        return self.controllerState

class DataTracker:
    def __init__(self):
        self.initialisedPrevState = False
        self.gameStates = []
        self.timeDifferences = []
        self.ctrlInputs = []
        self.dataCount = 0
        self.dataPerWrite = 1000
        self.writeCount = 0
        self.fileName = "MovementData/" + str(time.time()) + ".csv"
        self.generateFormatFile()



    def generateFormatFile(self):
      
        with open(self.fileName, 'w', newline='') as csvFile:
            dataFormatWriter = csv.writer(csvFile)
            gameStateHeaders = ['ballLocX', 'ballLocY', 'ballLocZ', 'ballVelX', 'ballVelY', 'ballVelZ', 'carLocX', 'carLocY', 'carLocZ', 'carPitch', 'carYaw', 'carRoll', 'carVelX', 'carVelY', 'carVelZ']
            controlHeaders = ["throttle", "steer", "time"]
            dataFormatHeader = []
            for h in gameStateHeaders:
                dataFormatHeader.append(h + "_0")
            for h in gameStateHeaders:
                dataFormatHeader.append(h + "_1")
            dataFormatHeader.extend(controlHeaders)
            dataFormatWriter.writerow(dataFormatHeader)


    def processState(self, deltaTime, prevContrState, newGameState):
        if not self.initialisedPrevState:
            self.gameStates.append(newGameState)
            self.initialisedPrevState=True
        else:
            self.gameStates.append(newGameState)
            self.timeDifferences.append(deltaTime)
            self.ctrlInputs.append(prevContrState)
            self.initialised = True
            self.dataCount += 1
        if (self.dataCount > 0) and (self.dataCount % self.dataPerWrite == 0):
            print("Have collected " + str(self.dataPerWrite) + " data points, writing to csv")
            sys.stdout.flush()
            dataOffset = self.writeCount*self.dataPerWrite
            self.writeCount += 1
            with open(self.fileName, 'a', newline='') as csvFile: #a - append
                movementWriter = csv.writer(csvFile)

                for i in range(dataOffset, self.dataCount):
                    dataUnit = DataUnit(self.gameStates[i], self.timeDifferences[i], self.gameStates[i+1], self.ctrlInputs[i])
                    movementWriter.writerow(dataUnit.getStrList())
                    #print(dataUnit.getStrList())
              

class GameState:
    def __init__(self, ballLoc, ballVel, carLoc, carOri, carVel):
        self.ballLoc = ballLoc
        self.ballVel = ballVel
        self.carLoc = carLoc
        self.carOri = carOri
        self.carVel = carVel

    def convertToStrList(self):
        gameList = []
        gameList.extend(self.ballLoc.getStrList())
        gameList.extend(self.ballVel.getStrList())
        gameList.extend(self.carLoc.getStrList())
        gameList.extend(self.carOri.getStrList())
        gameList.extend(self.carVel.getStrList())
        return gameList
        
class DataUnit:
    def __init__(self, prevGameState, deltaT=None, newGameState=None, ctrlInputs=None):
        if deltaT is None:
            #we only have one game state - the first row in the csv.
            prevStrList = prevGameState.convertToStrList()
            blankStrList = ["0"] * (len(prevStrList)+3) #3 for deltaT, throttle, steer
            self.strList = prevStrList + blankStrList
        else:
            #we have a full row of data
            prevStrList = prevGameState.convertToStrList()
            newStrList = newGameState.convertToStrList()
            thr = str(ctrlInputs[0])
            st = str(ctrlInputs[1])
            self.strList = prevStrList + newStrList + [thr] + [st] + [str(deltaT)]

    def getStrList(self):
        return self.strList


class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, val):
        return Vector3(self.x + val.x, self.y + val.y, self.z + val.z)

    def __sub__(self, val):
        return Vector3(self.x - val.x, self.y - val.y, self.z - val.z)

    def getStrList(self):
        return [str(self.x), str(self.y), str(self.z)]

    """def correction_to(self, ideal):
        # The in-game axes are left handed, so use -x
        current_in_radians = math.atan2(self.y, -self.x)
        ideal_in_radians = math.atan2(ideal.y, -ideal.x)

        correction = ideal_in_radians - current_in_radians

        # Make sure we go the 'short way'
        if abs(correction) > math.pi:
            if correction < 0:
                correction += 2 * math.pi
            else:
                correction -= 2 * math.pi

        return correction"""


"""def get_car_facing_vector(car):
    pitch = float(car.physics.rotation.pitch)
    yaw = float(car.physics.rotation.yaw)
    roll = float(car.physics.rotation.roll)

    facing_x = math.cos(pitch) * math.cos(yaw)
    facing_y = math.cos(pitch) * math.sin(yaw)

    return Vector2(facing_x, facing_y)"""
