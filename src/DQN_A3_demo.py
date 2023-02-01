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
from DQN_A3_train import Environment


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

def go_turn_left(rob): #3sec = 45 deg
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




def demo(env,rob,path_to_model):
    """
    :param env:
    :param rob:
    :param path_to_model: should contain the Q and T network.pt, example: "src/logs/task2/agent2/"
    :return:
    """

    discount_rate = 0.99
    buffer_size = 300
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 400
    lr = 0.001
    min_replay_size = 100

    device = 'cpu'
    dagent = DDQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size,min_replay_size,
                       cold_boot=True)
    dagent.load(path_to_model,device)

    obs = env.get_state()
    #RL hack
    ir = obs[:8]
    cam = obs[8:]
    #
    # cam[0] = cam[0]/1.2
    ir = ir/2
    prev_ir = ir
    obs = np.append(ir,cam)
    while True:
        # print(ir,"\n",cam)
        action = dagent.choose_action(0, obs, True)[0]
        take_action(action,rob)
        new_obs = env.get_state()

        ir = new_obs[:8]  # RL hack
        if np.equal(ir, prev_ir).all():
            ir = np.array([0.,0.,0.,0.,0.,0.,0.,0.])
        else :
            prev_ir = ir
        cam = new_obs[8:]
        # cam[0] = cam[0] / 1.2
        ir = ir / 2


        new_obs = np.append(ir, cam)
        obs = new_obs


def main():
    signal.signal(signal.SIGINT, terminate_program)
    rendering = True
    rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.2.53")
    env = Environment(only_front=False, rob = rob, rendering=rendering)
    rob.set_phone_tilt(108, 50)
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

    demo(env,rob,path_to_model="src/logs/task3/agent1/")
    # demo(env,rob,path_to_model="src/models/task3/")



if __name__ == "__main__":
    main()



