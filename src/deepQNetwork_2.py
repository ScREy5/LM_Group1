#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np
from numpy import inf

import torch
from collections import namedtuple
from architecture import DQN, ReplayMemory
import color_recog
import cv2
import random
import math
import matplotlib.pyplot as plt
import csv


import robobo
import sys
import signal

TRANSITION = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def turn_robobo(robo, repetition = 0, inverse = False, rotation_speed = 11, rotation_length = 1100):
    inversion = -1 if inverse else 1
    left, right = -rotation_speed * inversion, rotation_speed * inversion
    for i in range(repetition):
        robo.move(left, right, rotation_length)

def select_action(state, policy_network):
    # Will have to be changed
    global STEPS
    if eval_mode:
        with torch.no_grad():
            return policy_network(state).max(1)[1].view(1, 1)
    else:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * STEPS / EPS_DECAY)
        # TODO Will have to be changed. Step choice atm is arbitrary
        STEPS += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_network(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randint(0,2)]], dtype=torch.long)


def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_rewards(rewards_list, show_result=False):
    plt.figure(1)
    rewards_sum = torch.tensor(rewards_list, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_sum.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_sum) >= 100:
        means = rewards_sum.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model(memory, policy_network, target_network, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = TRANSITION(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_network(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)
    optimizer.step()


def read_input(rob):
    if simulation:
        X_input = np.log(np.array(rob.read_irs())) / -10
    else:
        X_input = np.log(np.array(rob.read_irs())) / 10
    X_input[X_input == -inf] = 0
    X_input[X_input == inf] = 0
    return X_input

def get_image_values(rob):
    # Following code gets an image from the camera
    image = rob.get_image_front()
    # IMPORTANT! `image` returned by the simulator is BGR, not RGB
    cv2.imwrite("afterstep_picture.png", image)
    green_pixel_ratio, center_x, center_y = color_recog.get_green_coord("afterstep_picture.png")
    return green_pixel_ratio, center_x, center_y

def environment_step_task1(rob, action, prev_reward):
    X_input = np.log(np.array(rob.read_irs())) / 10
    X_input[X_input == -inf] = 0
    # Normal scale
    # if X_input.min() <= -0.32
    # For scaled down
    if X_input.min() <= -0.6:
        reward = -1000
        done = True
        truncated = True
    elif action == 0 or action == 2:
        reward = -1 if prev_reward >= 0 else -1 + prev_reward
        X_input = np.log(np.array(rob.read_irs())) / 10
        X_input[X_input == -inf] = 0
        inverse = False if action == 0 else True
        turn_robobo(rob, 1, inverse)
        done = False
        truncated = True
    else:
        reward = 1 if prev_reward < 1 else 1 + prev_reward
        # Normal scale
        rob.move(10, 10, 1000)
        # Half scale
        # rob.move(15, 15, 1500)
        X_input = np.log(np.array(rob.read_irs())) / 10
        X_input[X_input == -inf] = 0
        done = False
        truncated = True

    return X_input, reward, done, truncated


def environment_step_task2(rob, action, green_ratio):
    global FOOD_CONSUMED
    if action == 0 or action == 2:
        reward = -1
        X_input = read_input(rob)
        new_green_ratio, centerx, centery = get_image_values(rob)
        output = np.append(X_input, [new_green_ratio, centerx, centery])
        inverse = False if action == 0 else True
        turn_robobo(rob, 1, inverse)
        done = False
        truncated = True
    else:
        reward = -1
        rob.move(10, 10, 1000)
        # Check if ratio increased
        X_input = read_input(rob)
        new_green_ratio, centerx, centery = get_image_values(rob)
        output = np.append(X_input, [new_green_ratio, centerx, centery])
        if new_green_ratio > green_ratio:
            reward = -0.5 + new_green_ratio
        # Else check if green ratio was big, as well as new ratio suddenly being smaller
        elif green_ratio > 0.33 and new_green_ratio < green_ratio:
            reward = 100
            FOOD_CONSUMED += 1
        done = False
        truncated = True

    if X_input.max() >= 0.4 and green_ratio < 0.33:
        reward = -500
        done = False
        truncated = True

    if FOOD_CONSUMED >= MAX_FOOD:
        done = True
    return output, reward, done, truncated

# Replace args with arg reader
def main():
    signal.signal(signal.SIGINT, terminate_program)


    # Setting up running environment
    if simulation:
        rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
        rob.play_simulation()

    else:
        rob = robobo.HardwareRobobo(camera=True).connect(address=ip_address)


    if eval_mode:
        policy_network = torch.load(model_path)
        policy_network.eval()
    else:
        policy_network = DQN.DQN(n_observations, n_actions, n_of_nodes)
        target_network = DQN.DQN(n_observations, n_actions, n_of_nodes)
        target_network.load_state_dict(policy_network.state_dict())

        optimizer = torch.optim.AdamW(policy_network.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory.ReplayMemory(10000)


    episode_durations = []
    reward_sums = []
    actions_taken_all = []

    for episode in range(NUM_EPISODES):
        print(f"Starting simulation #{episode}")
        # Set phone tilt to right value
        rob.set_phone_tilt(107.7, 50)
        # IMPORTANT! `image` returned by the simulator is BGR, not RGB
        if simulation:
            rob.play_simulation()
            X_input = np.log(np.array(rob.read_irs())) / -10
        else:
            X_input = np.log(np.array(rob.read_irs())) / 10
        X_input[X_input == -inf] = 0
        X_input[X_input == inf] = 0
        if task == 1:
            state = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0)
        if task == 2:
            new_green_ratio, centerx, centery = get_image_values(rob)
            state = torch.tensor(np.append(X_input, [new_green_ratio, centerx, centery]), dtype=torch.float32).unsqueeze(0)
        prev_reward = 0
        reward_total = 0
        actions_taken = []
        global FOOD_CONSUMED
        FOOD_CONSUMED = 0
        for t in range(steps_max):
            action = select_action(state, policy_network)
            actions_taken.append(action.item())
            if task == 1:
                observation, reward, terminated, truncated = environment_step_task1(rob, action.item(), prev_reward)
                reward = torch.tensor([reward])
                prev_reward = reward.item()
                reward_total += reward.item()
            elif task == 2:
                # Inefficient, takes photo multiple times instead of once
                green_ratio, centerx, centery = get_image_values(rob)
                observation, reward, terminated, truncated = environment_step_task2(rob, action.item(), green_ratio)
                reward += -(t/steps_max)
                reward = torch.tensor([reward])
                reward_total += reward.item()
            else:
                continue
            if terminated and not simulation:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            state = next_state

            if not eval_mode:
                memory.push(state, action, next_state, reward)
                optimize_model(memory, policy_network, target_network, optimizer)

                # Soft update of the target networks weights
                target_network_state_dict = target_network.state_dict()
                policy_network_state_dict = policy_network.state_dict()
                for key in policy_network_state_dict:
                    target_network_state_dict[key] = policy_network_state_dict[key] * TAU + target_network_state_dict[
                        key] * (1 - TAU)
                target_network.load_state_dict(target_network_state_dict)

            if t == steps_max - 1 or terminated:
                episode_durations.append(t+1)
                reward_sums.append(reward_total)
                actions_taken_all.append(actions_taken)
                plot_durations(episode_durations)
                plot_rewards(reward_sums)
                break

            # Break point if sensor reaches this point
            # if X_input.min() <= -0.30:
            # Standardised movement:
            #     rob.move(5, 5, 1000)

            # print("Base sensor detection: ", rob.base_detects_food())
        # pause the simulation and read the collected food
        if simulation:
            rob.pause_simulation()
            # Stopping the simulation resets the environment
            rob.stop_world()
        print(f"Stopping simulation #{episode}")
        time.sleep(5)

        #TODO checkpoint was 50
        if episode % checkpoint == 0 and not eval_mode:
            #"testing_results/target_network_full_v2_scaleChange_episode_"
            path1 = path_target_network + f"{episode}"
            # f"testing_results/policy_network_full_v2_scaleChange_episode_{episode}"
            path2 = path_policy_network + f"{episode}"
            torch.save(target_network, path1)
            torch.save(policy_network, path2)
        if save_action:
            with open(path_actions_taken, "w") as f:
                wr = csv.writer(f)
                wr.writerows(actions_taken_all)
    if not eval_mode:
        torch.save(target_network, target_network)
        torch.save(policy_network, policy_network)

if __name__ == "__main__":
    # Does this run in simulation?
    simulation = True
    # Is this run a validation / demo run?
    eval_mode = False
    # Ip address of robobo
    ip_address = "192.168.6.1"
    # Path of saved model
    model_path = "src/dQn_demo.py"
    # Number of episodes
    NUM_EPISODES = 500
    # Maximum number of steps
    steps_max = 250
    # How often will the model be saved (in episode number)
    checkpoint = 10
    # Will actions taken be saved as well
    save_action = False

    # Save location of target network during training
    path_target_network = "testing_results_task2/target_network_task2_episode_"
    path_policy_network = "testing_results_task2/policy_network_task2_episode_"
    path_actions_taken = "testing_results_task2/action_list"
    # Definition of DQN model
    n_observations = 11
    n_actions = 3
    n_of_nodes = 120

    # Definition of epsilon greedy values
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 2500

    # Optimization values
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.005
    LR = 1e-4

    # Total steps taken
    STEPS = 0
    task = 2
    FOOD_CONSUMED = 0
    MAX_FOOD = 7

    main()
