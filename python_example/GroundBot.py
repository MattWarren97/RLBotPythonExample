import math
import time
import csv
import random
import sys

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


def setInstrLength():
    return random.uniform(0, 3)

class GroundBot(BaseAgent):

    def initialize_agent(self):
        #This runs once before the bot starts up
        self.controllerState = SimpleControllerState() #to be returned by get_output on each tick
        self.currentGameState = None #game state in the most recent packet received.
        self.currentInstrLength = 0 #duration current instruction has been given.
        self.prevPacketSysTime = 0 #if instruction has changed, -- end time of previous instruction
        self.ticksPerInstr = 0 #number of physics ticks on current instructions.
        self.needNewInstr = False #true when a new instruction needs to be given
        self.instrStartTime = 0 #time when the new instruction was started
        self.instrLength = 0 #duration for which the instruction should be held.
        self.dataTracker = DataTracker() #handles writing the data

    def processTime(self):
        #First: check if the current instruction should have ended:
        newTime = time.clock()
        self.currentInstrLength = newTime - self.instrStartTime
        if self.currentInstrLength >= self.instrLength:
            self.needNewInstr = True
            #print("Need new instr true")

            self.instrStartTime = newTime
            self.instrLength = setInstrLength()
        else:
            self.ticksPerInstr += 1

    def processState(self):
        if self.needNewInstr:
            controls = [self.controllerState.throttle, self.controllerState.steer]
            self.dataTracker.processState(self.currentInstrLength, controls, self.currentGameState)
            print("Previous instructions have ended after ", self.currentInstrLength, " seconds, with controls: thr, st: ", controls[0], ", ", controls[1])
            self.setNewInstructions()


    def setNewInstructions(self):
        self.needNewInstr = False
        self.controllerState.throttle = (random.random()*2)-1 #between -1 and 1
        self.controllerState.steer = (random.random()*2)-1
        print("New instructions are throttle: " + str(self.controllerState.throttle) + ", steer: " + str(self.controllerState.steer))
        self.ticksPerInstr = 0

    def updateGameState(self, packet):
        ball_location = Vector3(packet.game_ball.physics.location.x, packet.game_ball.physics.location.y, packet.game_ball.physics.location.z)
        ball_velocity = Vector3(packet.game_ball.physics.velocity.x, packet.game_ball.physics.velocity.y, packet.game_ball.physics.velocity.z)
        my_car = packet.game_cars[self.index]
        car_location = Vector3(my_car.physics.location.x, my_car.physics.location.y, my_car.physics.location.z)
        car_orientation = Vector3(my_car.physics.rotation.pitch, my_car.physics.rotation.yaw, my_car.physics.rotation.roll)
        car_velocity = Vector3(my_car.physics.velocity.x, my_car.physics.velocity.y, my_car.physics.velocity.z)
        self.currentGameState = GameState(ball_location, ball_velocity, car_location, car_orientation, car_velocity)


    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        #this method is automatically called by the framework once on each physics tick.
        gameActive = packet.game_info.is_round_active
        if not gameActive:
            return self.controllerState
        self.updateGameState(packet)

        self.processTime() #determine if a new instruction is needed
        self.processState()
        

        return self.controllerState

class DataTracker:
    def __init__(self):
        self.prevGameState = None 
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


    def processState(self, instructionDuration, instructions, currentGameState):
        if self.prevGameState is not None: 
            #not the first call to this method, 'prev' variables are already initialised
            with open(self.fileName, 'a', newline='') as csvFile: #a - append
                movementWriter = csv.writer(csvFile)

                dataUnit = DataUnit(self.prevGameState, instructionDuration, currentGameState, instructions)
                movementWriter.writerow(dataUnit.getStrList())

        self.prevGameState = currentGameState

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
