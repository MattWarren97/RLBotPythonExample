import math
import time
import csv
import random
import sys

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket




class GroundBot(BaseAgent):

    def initialize_agent(self):
        #This runs once before the bot starts up
        self.controllerState = SimpleControllerState()
        self.prevInstr = self.controllerState
        self.prevPacketSysTime = 0
        self.deltaTime = 0;
        self.initialTime = 0
        self.ticksPerSecond = 0
        self.instrNewSecond = True
        self.dataTracker = DataTracker()

    #work out length of time since previous frame.
    def processTime(self):
        if (self.initialTime == 0):
            self.prevPacketSysTime = time.clock()
            self.initialTime = self.prevPacketSysTime
            self.secondStartTime = self.initialTime
            self.deltaTime = 0
            return False
        else:
            self.ticksPerSecond += 1
            newTime = time.clock()
            self.deltaTime = newTime-self.prevPacketSysTime
            self.prevPacketSysTime=newTime

            if newTime-self.secondStartTime >= 1:
                #1 second has passed
                print(str(self.ticksPerSecond) + " physics ticks in the last second")
                self.ticksPerSecond = 0
                self.instrNewSecond = True
                self.secondStartTime = newTime
            return True

    def processState(self, deltaTime, prevContrState, gameState):
        self.dataTracker.processState(deltaTime, prevContrState, gameState)

    def setInstructions(self, ballLoc, gameState):
        if self.instrNewSecond:
            self.instrNewSecond = False
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
        
        ball_location = Vector2(packet.game_ball.physics.location.x, packet.game_ball.physics.location.y)
        my_car = packet.game_cars[self.index]
        car_location = Vector2(my_car.physics.location.x, my_car.physics.location.y)
        car_direction = get_car_facing_vector(my_car)
        car_velocity = Vector2(my_car.physics.velocity.x, my_car.physics.velocity.y)
        
        gameActive = packet.game_info.is_round_active
        if not gameActive:
            return self.controllerState
        currentState = GameState(car_location, car_direction, car_velocity)

        self.processState(self.deltaTime, self.prevInstr, currentState)
        
        self.setInstructions(ball_location, currentState)
        

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
        self.initialised = False
        self.gameStates = []
        self.timeDifferences = []
        self.ctrlInputs = []
        self.dataCount = 0

    def processState(self, deltaTime, prevContrState, newGameState):
        if not self.initialised:
            self.gameStates.append(newGameState)
            self.initialised=True
        else:
            self.gameStates.append(newGameState)
            self.timeDifferences.append(deltaTime)
            self.ctrlInputs.append(prevContrState)
            self.initialised = True
            self.dataCount += 1
            if (self.dataCount%1000 == 0):
                print("Data count is " + str(self.dataCount))
        if (self.dataCount > 0) and (self.dataCount % 10000 == 0):
            print("Have collected 10000 data points, writing to csv")
            sys.stdout.flush()
            with open('MovementData.csv', 'w') as csvFile:
                movementWriter = csv.writer(csvFile)
                firstDataUnit = DataUnit(self.gameStates[0])
                movementWriter.writerow(firstDataUnit.getStrList())
                for i in range(0, self.dataCount):
                    dataUnit = DataUnit(self.gameStates[i], self.timeDifferences[i], self.gameStates[i+1], self.ctrlInputs[i])
                    movementWriter.writerow(dataUnit.getStrList())
              

class GameState:
    def __init__(self, carLoc, carDir, carVel):
        self.carLoc = carLoc
        self.carDir = carDir
        self.carVel = carVel

    def convertToStrList(self):
        return [str(self.carLoc.x), str(self.carLoc.y), str(self.carDir.x), str(self.carDir.y), str(self.carVel.x), str(self.carVel.y)]

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
            thr = str(ctrlInputs.throttle)
            st = str(ctrlInputs.steer)
            self.strList = prevStrList + newStrList + [thr] + [st] + [str(deltaT)]

    def getStrList(self):
        return self.strList


class Vector2:
    def __init__(self, x=0, y=0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, val):
        return Vector2(self.x + val.x, self.y + val.y)

    def __sub__(self, val):
        return Vector2(self.x - val.x, self.y - val.y)

    def correction_to(self, ideal):
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

        return correction


def get_car_facing_vector(car):
    pitch = float(car.physics.rotation.pitch)
    yaw = float(car.physics.rotation.yaw)

    facing_x = math.cos(pitch) * math.cos(yaw)
    facing_y = math.cos(pitch) * math.sin(yaw)

    return Vector2(facing_x, facing_y)
