#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
from DQN_Agent2 import *
import os
import vrep
import color_recog

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
        self.food = 0
        self.got_closer = False

    def reset(self):
        # pause the simulation and read the collected food
        self.rob.pause_simulation()
        time.sleep(2)
        # Stopping the simulation resets the environment
        self.rob.stop_world()
        self.rob.wait_for_stop()
        vrep.simxSetBooleanParameter(self.rob._clientID, vrep.sim_boolparam_display_enabled, self.rendering,
                                         vrep.simx_opmode_oneshot)
        self.rob.play_simulation()
        self.rob.set_phone_tilt(107.8, 50)
        self.time = 0
        self.food = 0
        self.last_green = 0
        self.terminated = False
        self.past_actions = []
        self.got_closer = False
        return self.get_state()

    def get_state(self):
        if self.front_only:
            obs = continuous_state(read_ir(self.rob))[3:]
        else:
            obs = continuous_state(read_ir(self.rob))
        obs = np.append(obs,np.array(get_image_values(self.rob)))
        return obs


    def get_reward(self,state):
        ir = state[:8]
        green = state[-3]

        reversed_actions = self.past_actions[::-1] #reverse past actions
        if np.any([ir>=0.31]) and self.food == self.rob.collected_food(): #check if we observed touch with wall
            self.terminated = True
            reward = -200
        else:
            last_action = reversed_actions[0] #get last action

            if last_action == 1 and green > self.last_green: #if straight and closer
                reward = -1+green
                # self.got_closer = True
            elif last_action == 1 and self.rob.collected_food() > self.food: #if straight and got food
                reward = 100
            elif last_action == 1:
                reward = -2
            else : #if turned
                reward = -1
            #Experimental
            # if self.got_closer and green > 0.1:
            #     reward += 1
            #     self.got_closer = False

        if self.time == 300 or self.food == 7:
            self.terminated = True

        self.food = self.rob.collected_food()
        self.last_green = green
        return reward

    def step(self,action):
        take_action(action,self.rob)
        self.time += 1
        self.past_actions.append(action)
        state = self.get_state()
        reward = self.get_reward(state)
        return state, reward, self.terminated

def get_image_values(rob):
    # Following code gets an image from the camera
    image = rob.get_image_front()
    # IMPORTANT! `image` returned by the simulator is BGR, not RGB
    cv2.imwrite("afterstep_picture.png", image)
    green_pixel_ratio, center_x, center_y = color_recog.get_green_coord("afterstep_picture.png")
    return green_pixel_ratio, center_x, center_y
def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def read_ir(rob):
    br, bc, bl, frr, fr, fc, fl, fll = np.log(np.array(rob.read_irs())) / -10
    return br, bc, bl, frr, fr, fc, fl, fll

def continuous_state(vals):
    vals = np.abs(np.array(vals))
    vals[vals == np.inf] = 0
    return vals

def go_turn_left(rob):
    rob.move(-11,11,550)

def go_turn_right(rob):
    rob.move(11,-11,550)

def go_straight(rob):
    rob.move(15,15,1500)

def take_action(action,rob):
    if action == 1:
        go_straight(rob)
    if action == 0:
        go_turn_left(rob)
    if action == 2:
        go_turn_right(rob)


def train(env,path_to_save,path_to_load = None):


    #Hyperparams to tune
    # More or less fixed
        # Starting value of epsilon
    epsilon_start = 1.0
        # End value (lowest value) of epsilon
    epsilon_end = 0.05
        # Discount rate
    discount_rate = 0.99
        # That is the sample that we consider to update our algorithm each step
    batch_size = 32
        #Maximum Episodes
    max_episodes = 100
    """ Needs to be tuned/adjusted """
        # Maximum number of transitions that we store in the buffer
    buffer_size = 200
        # Minimum number of transitions that we need to initialize
    min_replay_size = 50
        # Decay period until epsilon start -> epsilon end in steps
    epsilon_decay = 1500
        # Learning_rate
    lr = 0.0001
        # Update frequency of the target network in steps
    update_freq = 25
        # Autosaving frequency int num (set to None if you don't want to save)
    auto_save = 1


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dagent = DDQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size, min_replay_size)

    if not path_to_load == None:
        dagent.load(path_to_load)
        steps_taken = np.loadtxt(path_to_load+"all_steps.txt")
        status = [steps_taken,int(path_to_load[-2:])]
    else:
        status = None

    training_loop(env, dagent, max_episodes, batch_size= batch_size,target_ = True, auto_save= auto_save, cont = status, update_freq = update_freq)
    dagent.save(path_to_save)

def validate(env,path_to_model,path_to_save):

    #Placeholders needed for agent
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
    dagent.load(path_to_model,device)
    terminated = False
    obs = env.reset()
    while not terminated:
        action = dagent.choose_action(0, obs, True)[0]
        new_obs, rew, terminated = env.step(action)
        obs = new_obs

    if path_to_save != None:
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        np.savetxt(path_to_save + "val_dq_.txt", env.past_actions)


def main():
    signal.signal(signal.SIGINT, terminate_program)
    rendering = True
    rob = robobo.SimulationRobobo("").connect(address='145.108.232.183', port=19997, rendering=rendering)  # local IP needed
    env = Environment(only_front=False,rob = rob, rendering=rendering)
    rob.play_simulation()
    rob.set_phone_tilt(107.8, 50)

    """ This Segment is for playing and testing in sim!"""
    # while True:
        # print(get_image_values(rob))
        # print(env.get_state())
        # quit()
    #     rob.move(20,20,1000)
    #     print(rob.collected_food())
    #     if rob.collected_food() >= 2:
    #         rob.pause_simulation()
    #         rob.stop_world()
    #         rob.wait_for_stop()
    #         rob.play_simulation()

    # print(get_image_values(rob))

    """ Actual Training """
    train(env,"logs/task2/agent2/", path_to_load=None)
    # validate(env,"models/task2_wednesday/agent2/",None)




if __name__ == "__main__":
    main()



