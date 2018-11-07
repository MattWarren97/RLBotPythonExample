import math
import time
import csv
import random
import sys


#if errors on imports - eg (import tensorflow as tf)
#may need to include the imports INSIDE the class
#Saltie discussed it on the Discord ml-discussion channel 7thAugust2018
#linked to: https://github.com/SaltieRL/Saltie/blob/Dragon/main_agent/saltie.py

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator



def getRandInstrLength():
    return random.uniform(0, 3)

class GroundBot(BaseAgent):

    def initialize_agent(self):

        from GroundLearner import GroundLearner
        #this also has to be imported in here. Cool eh!


        #This runs once before the bot starts up
        self.controllerState = SimpleControllerState() #to be returned by get_output on each tick
        self.currentGameModel = None #game state in the most recent packet received.
        self.currentInstrLength = 0 #duration current instruction has been given.
        self.prevPacketSysTime = 0 #if instruction has changed, -- end time of previous instruction
        self.ticksPerInstr = 0 #number of physics ticks on current instructions.
        self.needNewInstr = False #true when a new instruction needs to be given
        self.instrStartTime = 0 #time when the new instruction was started
        self.instrLength = 0 #duration for which the instruction should be held.
        self.dataTracker = DataTracker() #handles writing the data
        
        movementData = "MovementData/"
        self.learner = GroundLearner(movementData)
        #self.movementMLP = self.learner.trainMLPRegressor()
        self.learner.trainHitBallMLP()
        self.hitBall = True
        self.ballReset = False

        
        #fancy tf stuff needed for generating truncated normal values
        #self.tfSession = tf.Session()
        #self.t_normal_gen = tf.truncated_normal((2,), mean=0, stddev=0.5)
        

    def setInstructionTime(self, instrTime):
        self.instrStartTime = time.clock()
        self.instrLength = instrTime

    def processTime(self, packet):
        #First: check if the current instruction should have ended:
        newTime = time.clock()
        self.currentInstrLength = newTime - self.instrStartTime
        if self.currentInstrLength >= self.instrLength:
            self.needNewInstr = True
            self.setInstructionTime(getRandInstrLength())

            #self.instrStartTime = newTime
            #self.instrLength = setInstrLength()
        elif self.hitBall:
            if self.currentInstrLength >= self.instrLength-1 and not self.ballReset:
                #within last second, reset the ball position.
                self.resetBall(packet)
            self.ticksPerInstr += 1

    def processState(self):
        if self.needNewInstr:
            controls = [self.controllerState.throttle, self.controllerState.steer]
            self.dataTracker.processState(self.currentInstrLength, controls, self.currentGameModel)
            print("Previous instructions have ended after ", self.currentInstrLength, " seconds, with controls: thr, st: ", controls[0], ", ", controls[1])
            if self.hitBall:
                self.setHitBallInstructions()
            else:
                self.setRandInstructions()


    
    def updateGameModel(self, packet):
        my_car = packet.game_cars[self.index]
        car_location = V3(my_car.physics.location.x, my_car.physics.location.y, my_car.physics.location.z)
        car_orientation = V3(my_car.physics.rotation.pitch, my_car.physics.rotation.yaw, my_car.physics.rotation.roll)
        car_velocity = V3(my_car.physics.velocity.x, my_car.physics.velocity.y, my_car.physics.velocity.z)
        ball_location = V3(packet.game_ball.physics.location.x, packet.game_ball.physics.location.y, packet.game_ball.physics.location.z)
        ball_velocity = V3(packet.game_ball.physics.velocity.x, packet.game_ball.physics.velocity.y, packet.game_ball.physics.velocity.z)
        self.currentGameModel = GameModel(ball_location, ball_velocity, car_location, car_orientation, car_velocity)



    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        #this method is automatically called by the framework once on each physics tick.
        gameActive = packet.game_info.is_round_active
        if not gameActive:
            return self.controllerState
        self.updateGameModel(packet)

        self.processTime(packet) #determine if a new instruction is needed
        self.processState()
        

        return self.controllerState

    def setHitBallInstructions(self):
        #build a feature: ballPos, ballVel, carPos, carOri, carVel, predictedBallPos
        self.needNewInstr = False
        self.ballReset = False

        gm = self.currentGameModel
        features = []
        predBallLoc, predBallVel = self.predictBallState()
        for o in [gm.carLoc, gm.carOri, gm.carVel, predBallLoc]:
            features.extend(o.getStrList())

        features = [features]  #need to input a 2D array, even with only one record.
        print("Hit-ball features are: ", features)
        scaledF = self.learner.hitBallScaler.transform(features)
        print("Scaled hit-ball features are: ", scaledF)

        targets = self.learner.hbMLP.predict(scaledF)
        print("New Hit-Ball Instruction: ", targets)
        print(flush=True)
        self.controllerState.throttle = targets[0][0]
        self.controllerState.steer = targets[0][1]
        self.setInstructionTime(targets[0][2]+2)



    def setRandInstructions(self):
        #want random throttle and steer; want to aim for steer usually close to 0, throttle usually close to -1 or 1
        self.needNewInstr = False
        #with self.tfSession.as_default():
        #    (notThrottle, steer) = t_normal_gen.eval()
        #    if notThrottle < 0:
        #        self.controllerState.throttle = -1 - notThrottle
        #    else:
        #        self.controllerState.throttle = 1-notThrottle

        #    self.controllerState.steer = steer
        randomForThr = (random.random()*2)-1
        randomForSte = (random.random()*2)-1
        if randomForThr < 0:
            self.controllerState.throttle = -1 - (-1*(math.pow(randomForThr, 2)))
        else:
            self.controllerState.throttle = 1 - (math.pow(randomForThr, 2))
        if randomForSte < 0:
            self.controllerState.steer = -1*(math.pow(randomForSte, 2))
        else:
            self.controllerState.steer = (math.pow(randomForSte, 2))

        #self.controllerState.throttle = math.pow((random.random()*2)-1, 2) #between -1 and 1
        #self.controllerState.steer = (random.random()*2)-1
        print("New instructions are throttle: " + str(self.controllerState.throttle) + ", steer: " + str(self.controllerState.steer))
        self.ticksPerInstr = 0

    def predictBallState(self):
        return self.currentGameModel.ballLoc, self.currentGameModel.ballVel #for now, ball is static.

    def resetBall(self, packet):
        #to be used when the ball has been hit
        #forces the internal GameState to set the ball back static in mid-pitch.
        print("Would reset the ball, but doing nothing for now")
        self.ballReset = True

        #for some reason all this code is throwing errors on Vector3 has no attribute 'convert_to_flat'
        """
        my_car = packet.game_cars[self.index]
        car_state = CarState(
            jumped=my_car.jumped, 
            double_jumped=my_car.double_jumped,
            boost_amount=my_car.boost,
            physics = Physics(
                velocity = my_car.physics.velocity,
                rotation = my_car.physics.rotation,
                angular_velocity = my_car.physics.angular_velocity,
                location = my_car.physics.location)
            )#all set to their current values, as per the packet.

        ball_state = BallState()

        game_state = GameState(ball=ball_state, cars={self.index: car_state})

        self.set_game_state(game_state)
        """

class DataTracker:
    def __init__(self):
        self.prevGameModel = None 
        self.fileName = "MovementData/" + str(time.time()) + ".csv"
        self.generateFormatFile()

    def generateFormatFile(self):
      
        with open(self.fileName, 'w', newline='') as csvFile:
            dataFormatWriter = csv.writer(csvFile)
            gameModelHeaders = ['ballLocX', 'ballLocY', 'ballLocZ', 'ballVelX', 'ballVelY', 'ballVelZ', 'carLocX', 'carLocY', 'carLocZ', 'carPitch', 'carYaw', 'carRoll', 'carVelX', 'carVelY', 'carVelZ']
            controlHeaders = ["throttle", "steer", "time"]
            dataFormatHeader = []
            for h in gameModelHeaders:
                dataFormatHeader.append(h + "_0")
            for h in gameModelHeaders:
                dataFormatHeader.append(h + "_1")
            dataFormatHeader.extend(controlHeaders)
            dataFormatWriter.writerow(dataFormatHeader)


    def processState(self, instructionDuration, instructions, currentGameModel):
        if self.prevGameModel is not None: 
            #not the first call to this method, 'prev' variables are already initialised
            with open(self.fileName, 'a', newline='') as csvFile: #a - append
                movementWriter = csv.writer(csvFile)

                dataUnit = DataUnit(self.prevGameModel, instructionDuration, currentGameModel, instructions)
                movementWriter.writerow(dataUnit.getStrList())

        self.prevGameModel = currentGameModel

class GameModel:
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
    def __init__(self, prevGameModel, deltaT=None, newGameModel=None, ctrlInputs=None):
        prevStrList = prevGameModel.convertToStrList()
        newStrList = newGameModel.convertToStrList()
        thr = str(ctrlInputs[0])
        st = str(ctrlInputs[1])
        self.strList = prevStrList + newStrList + [thr] + [st] + [str(deltaT)]

    def getStrList(self):
        return self.strList

#vector3, but renamed to not clash with the other v3 used in internal game state.
class V3:
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, val):
        return V3(self.x + val.x, self.y + val.y, self.z + val.z)

    def __sub__(self, val):
        return V3(self.x - val.x, self.y - val.y, self.z - val.z)

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
