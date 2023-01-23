import numpy as np
import time
import vrep
class Environment():

    def __init__(self, only_front = True, rob = None, rendering = True):
        self.time = 0
        self.action_space = [0,1,2]
        self.past_actions = []
        self.terminated = False
        self.front_only = only_front
        self.observation_space_size = 5 if self.front_only else 8
        self.rob = rob
        self.rendering = rendering

    def reset(self):
        # pause the simulation and read the collected food
        self.rob.pause_simulation()
        # Stopping the simualtion resets the environment
        self.rob.stop_world()
        time.sleep(5)
        vrep.simxSetBooleanParameter(self.rob._clientID, vrep.sim_boolparam_display_enabled, self.rendering,
                                         vrep.simx_opmode_oneshot)
        self.rob.play_simulation()
        self.time = 0
        self.terminated = False
        self.past_actions = []
        # self.rob.move(0,0,100)
        return self.get_state()

    def get_state(self):
        if self.front_only:
            return continuous_state(read_ir(self.rob))[3:]
        else:
            return continuous_state(read_ir(self.rob))


    def get_reward(self,state):
        exponent = 0
        reversed_actions = self.past_actions[::-1] #reverse past actions
        # print(reversed_actions)
        if np.any([state>3.3]): #check if we observed touch
            self.terminated = True
            reward = -1000
        else:
            first_elem = reversed_actions[0] #get last action
            for i in range(len(reversed_actions)):
                if reversed_actions[i] == first_elem:
                    exponent += 1
                else: break

            if first_elem == 1: #if straight
            #     # reward =  abs(exponent*2)
                reward = abs(exponent)
            else: #if turned
                reward = -abs(exponent)

        # reward += np.sum(np.abs(state-5))/10 # the further you are the higher reward
        if self.time == 1000:
            reward += 1000
            self.terminated = True
        return reward

    def step(self,action):
        take_action(action,self.rob)
        self.time += 1
        self.past_actions.append(action)
        state = self.get_state()
        reward = self.get_reward(state)

        return state, reward, self.terminated

def read_ir(rob):
    br, bc, bl, frr, fr, fc, fl, fll = np.log(np.array(rob.read_irs())) / -1
    return br, bc, bl, frr, fr, fc, fl, fll

def discrete_state(vals,touch=3):
    # below low = nothing
    # low - mid = close
    # mid - high = touch
    # above high = nothing
    vals = np.digitize(vals,[1,1.5,2,2.5,touch,10])
    vals[vals > 5] = 0
    return vals

def continuous_state(vals):
    vals = np.array(vals)
    vals[vals > 100] = 0
    return vals

def go_turn_left(rob): #3sec = 45 deg
    rob.move(-11,11,1100)

def go_turn_right(rob):
    rob.move(11,-11,1100)

def go_straight(rob):
    rob.move(15,15,1500)

def take_action(action,rob):
    if action == 1:
        go_straight(rob)
    if action == 0:
        go_turn_left(rob)
    if action == 2:
        go_turn_right(rob)