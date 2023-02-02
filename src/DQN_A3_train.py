#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
from DQN_Agent3 import *
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
        self.got_red = False
        self.got_green = False
        self.last_green = 0
        self.last_red = 0
        self.red_trigger = False

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
        self.rob.set_phone_tilt(107.7, 50)
        self.time = 0
        self.last_green = 0
        self.last_red = 0
        self.terminated = False
        self.past_actions = []
        self.got_closer = False
        self.got_red = False
        self.got_green = False
        self.red_trigger = False
        return self.get_state()

    def get_state(self):
        if self.front_only:
            obs = continuous_state(read_ir(self.rob))[3:]
        else:
            obs = continuous_state(read_ir(self.rob))
        obs = np.append(obs,np.array(get_green_values(self.rob)))
        obs = np.append(obs, np.array(get_red_values(self.rob)))
        return obs



    def get_reward(self,state):
        self.got_red = self.rob.grabbed_food()
        ir = state[:8]
        green = state[-6]
        green_cent = state[-5]
        red = state[-3]
        if 0.00001 > red:
            red = 0
        red_cent = state[-2]
        f_rl = np.array([ir[4],ir[6]])
        f_rrll = np.array([ir[3],ir[7]])
        back = np.array(ir[:2])
        last_action = self.past_actions[-1]
        if np.any([back>=0.31]) or np.any([f_rl>=0.24]) or np.any([f_rrll>=0.22]): #collission check
            self.terminated = True
            reward = -300
        else:
            if not self.got_red:
                if red > 0 :
                    reward = -(abs(64-red_cent)/48)**2
                    if 20 > abs(64-red_cent) and last_action == 1:
                        reward += 3
                else :
                    if last_action == 1 and red == 0:
                        reward = -3
                    elif last_action != 1 and red == 0:
                        reward = -2
            else:
                if not self.red_trigger:
                    reward = 100
                else:
                    if not self.rob.base_detects_food():
                        if green > 0:
                            reward = -(abs(64 - green_cent) / 48) ** 2
                            if 20 > abs(64 - green_cent) and last_action == 1:
                                reward += 2
                        else:
                            if last_action == 1 and green == 0:
                                reward = -3
                            elif last_action != 1 and green == 0:
                                reward = -2
                    else :
                        reward = 500
                        self.terminated = True



        if self.got_red:
            self.red_trigger = True
        else:
            self.red_trigger = False


        if self.time == 300:
            self.terminated = True

        self.last_red = red
        self.last_green = green
        return reward

    def step(self,action):
        take_action(action,self.rob)
        self.time += 1
        self.past_actions.append(action)
        state = self.get_state()
        reward = self.get_reward(state)
        return state, reward, self.terminated

def get_green_values(rob):
    # Following code gets an image from the camera
    image = rob.get_image_front()
    # IMPORTANT! `image` returned by the simulator is BGR, not RGB
    cv2.imwrite("afterstep_picture.png", image)
    green_pixel_ratio, center_x, center_y = color_recog.get_green_coord("afterstep_picture.png")
    return green_pixel_ratio, center_x, center_y

def get_red_values(rob):
    # Following code gets an image from the camera
    image = rob.get_image_front()
    # IMPORTANT! `image` returned by the simulator is BGR, not RGB
    cv2.imwrite("afterstep_picture.png", image)
    green_pixel_ratio, center_x, center_y = color_recog.get_red_coord("afterstep_picture.png")
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
    rob.move(-5,5,550)

def go_turn_right(rob):
    rob.move(5,-5,550)

def go_straight(rob):
    rob.move(15,15,1000)

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
    discount_rate = 0.25
        # That is the sample that we consider to update our algorithm each step
    batch_size = 32
        #Maximum Episodes
    max_episodes = 300
    """ Needs to be tuned/adjusted """
        # Maximum number of transitions that we store in the buffer
    buffer_size = 100
        # Minimum number of transitions that we need to initialize
    min_replay_size = 100
        # Decay period until epsilon start -> epsilon end in steps
    epsilon_decay = 2000
        # Learning_rate
    lr = 0.0001
        # Update frequency of the target network in steps
    update_freq = 10
        # Autosaving frequency int num (set to None if you don't want to save)
    auto_save = 5


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dagent = DDQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size, min_replay_size)

    if not path_to_load == None:
        dagent.load(path_to_load,device)
        steps_taken = np.loadtxt(path_to_load+"all_steps.txt")
        status = [steps_taken,int(path_to_load[-3:-1])]
    else:
        status = None

    training_loop(env,
                  dagent,
                  max_episodes,
                  batch_size= batch_size,
                  target_ = True,
                  auto_save= auto_save,
                  cont = status,
                  update_freq = update_freq,
                  verbose=True,
                  auto_path=path_to_save)
    dagent.save(path_to_save)

def validate(env,path_to_model,path_to_save):


    for run in range(10):
        #Placeholders needed for agent
        discount_rate = 0.99
        buffer_size = 300
        epsilon_start = 1.0
        epsilon_end = 0.05
        epsilon_decay = 400
        lr = 0.001
        min_replay_size = 200


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dagent = DDQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size,min_replay_size,
                           cold_boot=True)
        dagent.load(path_to_model,device)
        terminated = False
        obs = env.reset()

        got_red = False
        got_green = False

        while not terminated:
            action = dagent.choose_action(0, obs, True)[0]
            new_obs, rew, terminated = env.step(action)
            obs = new_obs
            if env.got_red :
                got_red = True
            if env.rob.base_detects_food():
                got_green = True


        if path_to_save != None:
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            np.savetxt(path_to_save + f"run_{run+1}_val_result.txt", np.array([got_red,got_green]))


def main():
    signal.signal(signal.SIGINT, terminate_program)
    rendering = True
    rob = robobo.SimulationRobobo("").connect(address='127.0.0.1', port=19997, rendering=rendering)  # local IP needed
    env = Environment(only_front=False,rob = rob, rendering=rendering)
    rob.play_simulation()
    rob.set_phone_tilt(107.8, 50)

    """ This Segment is for playing and testing in sim!"""
    # while True:
    #     print(rob.grabbed_food())
    #     print(rob.base_detects_food())
    #     time.sleep(1)
        # print(read_ir(rob))
        # for _ in range(5):
        #     take_action(1,rob)
        # for _ in range(5):
        #     take_action(2,rob)
        # print(get_green_values(rob))
        # print(get_red_values(rob))
        # print(read_ir(rob))
        # time.sleep(1)
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
    # train(env,"logs/task3/agent3/", path_to_load=None)

    # unique_id = "Training" #so we don't save at the same place
    # arena ="Val4" #Train or Val1/Val2/Val3/Val4
    #
    # checkpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # for checkpoint in checkpoints:
    #     validate(env,f"models/task2_wed_night1/backup/autosave_{checkpoint}/", f"logs/task2/validation_runs/{unique_id}/{arena}_{checkpoint}/")
    validate(env,"logs/task3/agent3/",None)



if __name__ == "__main__":
    main()



