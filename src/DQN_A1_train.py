#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
from DQN_Agent1 import *
import os
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


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

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


def train_own(env,path_to_save,path_to_load = None):
    # Set the hyperparameters

    # Discount rate
    discount_rate = 0.99
    # That is the sample that we consider to update our algorithm
    batch_size = 32
    # Maximum number of transitions that we store in the buffer
    buffer_size = 200
    # Minimum number of transitions that we need to initialize
    min_replay_size = 100
    # Starting value of epsilon
    epsilon_start = 1.0
    # End value (lowest value) of epsilon
    epsilon_end = 0.05
    # Decay period until epsilon start -> epsilon end in steps
    epsilon_decay = 6000
    max_episodes = 500
    # Learning_rate
    lr = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dagent = DDQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size, min_replay_size)
    if not path_to_load == None:
        dagent.load(path_to_load)
        steps_taken = np.loadtxt(path_to_load+"all_steps.txt")
        steps_taken = [steps_taken,200]
    average_rewards_ddqn = training_loop(env, dagent, max_episodes, batch_size= batch_size,target_ = True, auto_save= True, cont = steps_taken)
    dagent.save(path_to_save)
    print(average_rewards_ddqn)

def validate_own(env,path_to_model,path_to_save):

    discount_rate = 0.99
    batch_size = 32
    buffer_size = 300
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 400
    max_episodes = 500
    lr = 0.001
    min_replay_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dagent = DDQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size,min_replay_size,
                       cold_boot=True)
    dagent.load(path_to_model)
    terminated = False
    obs = env.reset()
    step = 0
    while not terminated:
        # print(dagent.return_q_value(obs)[1].tolist())
        # action = np.random.choice(env.action_space)
        action = dagent.choose_action(0, obs, True)[0]
        new_obs, rew, terminated = env.step(action)
        obs = new_obs
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    np.savetxt(path_to_save + "val_dq_.txt", env.past_actions)

def main():
    signal.signal(signal.SIGINT, terminate_program)
    rendering = True
    rob = robobo.SimulationRobobo("#0").connect(address='192.168.1.142', port=19997, rendering=rendering)  # local IP needed
    env = Environment(only_front=False,rob = rob, rendering=rendering)
    rob.play_simulation()

    # # # Following code moves the robot
    # while True:
    #     print(read_ir(rob))
    #     print(np.array(rob.read_irs()))
    #     rob.move(3,3)
    #     action = np.random.choice(env.action_space)
    #     action = 0
    #     state, reward, terminated = env.step(action)
    #     print(f"{action=}, {state=}, {reward=}, {terminated=}")
    #     if terminated:
    #        env.reset()
    # pause the simulation and read the collected food
    # start = time.time()
    # for _ in range(10):
    #     print(rob.read_irs())
    #     rob.move(10,10,1000)
    # end = start - time.time()
    # print(end)
    # rob.pause_simulation()

    # Stopping the simualtion resets the environment
    # rob.stop_world()


    # train_own(env,"logs/base/","backup/autosave_200/")
    validate_own(env,"backup/autosave_0/","logs/dqn1/")



if __name__ == "__main__":
    main()



