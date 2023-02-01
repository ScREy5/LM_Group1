import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import os

seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class DQN(nn.Module):

    def __init__(self, env, learning_rate):
        '''
        Params:
        env = environment that the agent needs to play
        learning_rate = learning rate used in the update

        '''

        super(DQN, self).__init__()
        input_features = env.observation_space_size+6
        action_space = 3

        '''
        ToDo: 
        Write the layers of your neural network! 
        Make sure that the input features and the output features are in line with the environment that 
        the class takes as an input feature
        '''
        # Solution:

        self.dense1 = nn.Linear(in_features=input_features, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=32)
        self.dense3 = nn.Linear(in_features=32, out_features=16)
        self.dense4 = nn.Linear(in_features=16, out_features=action_space)

        # Here we use ADAM, but you could also think of other algorithms such as RMSprob
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        '''
        Params:
        x = observation
        '''

        '''
        ToDo: 
        Write the forward pass! You can use any activation function that you want (ReLU, tanh)...
        Important: We want to output a linear activation function as we need the q-values associated with each action

        '''

        # Solution:
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = self.dense4(x)

        return x


class ExperienceReplay:

    def __init__(self, env, buffer_size, min_replay_size=100):

        '''
        Params:
        env = environment that the agent needs to play
        buffer_size = max number of transitions that the experience replay buffer can store
        min_replay_size = min number of (random) transitions that the replay buffer needs to have when initialized
        seed = seed for random number generator for reproducibility
        '''
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([], maxlen=100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('Please wait, the experience replay buffer will be filled with random transitions')

        obs = self.env.reset()
        for _ in range(self.min_replay_size):
            '''
            ToDo: 
            Write a for loop that initializes the experience replay buffer with random transitions 
            such that the experience replay buffer 
            has minimum random transitions already stored 
            '''

            # Solution:
            action = np.random.choice(env.action_space)
            new_obs, rew, terminated = env.step(action)
            done = terminated

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                obs = env.reset()

        print('Initialization with random transitions is done!')

    def add_data(self, data):
        '''
        Params:
        data = relevant data of a transition, i.e. action, new_obs, reward, done
        '''
        self.replay_buffer.append(data)

    def sample(self, batch_size):

        '''
        Params:
        batch_size = number of transitions that will be sampled

        Returns:
        tensor of observations, actions, rewards, done (boolean) and next observation
        '''

        transitions = random.sample(self.replay_buffer, batch_size)

        # Solution
        observations = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = np.asarray([t[4] for t in transitions])

        # PyTorch needs these arrays as tensors!, don't forget to specify the device! (cpu / GPU)
        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32, device=self.device)

        return observations_t, actions_t, rewards_t, dones_t, new_observations_t

    def add_reward(self, reward):

        '''
        Params:
        reward = reward that the agent earned during an episode of a game
        '''

        self.reward_buffer.append(reward)


class DDQNAgent:

    def __init__(self, env, device, epsilon_decay,
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, min_replay_size, cold_boot = False):
        '''
        Params:
        env = name of the environment that the agent needs to play
        device = set up to run CUDA operations
        epsilon_decay = Decay period until epsilon start -> epsilon end
        epsilon_start = starting value for the epsilon value
        epsilon_end = ending value for the epsilon value
        discount_rate = discount rate for future rewards
        lr = learning rate
        buffer_size = max number of transitions that the experience replay buffer can store
        cold_boot = for inference MemoryReplay not needed
        '''
        self.env = env
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        self.losses = []

        if not cold_boot:
            self.replay_memory = ExperienceReplay(self.env, self.buffer_size, min_replay_size)
        self.online_network = DQN(self.env, self.learning_rate).to(self.device)

        '''
        ToDo: Add here a target network and set the parameter values to the ones of the online network!
        Hint: Use the method 'load_state_dict'!
        '''

        # Solution:
        self.target_network = DQN(self.env, self.learning_rate).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

    def choose_action(self, step, observation, greedy=False):

        '''
        Params:
        step = the specific step number
        observation = observation input
        greedy = boolean that

        Returns:
        action: action chosen (either random or greedy)
        epsilon: the epsilon value that was used
        '''

        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        random_sample = random.random()

        if (random_sample <= epsilon) and not greedy:
            # Random action
            action = np.random.choice(self.env.action_space)

        else:
            # Greedy action
            obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            q_values = self.online_network(obs_t.unsqueeze(0))

            max_q_index = torch.argmax(q_values, dim=1)[0]
            action = max_q_index.detach().item()

        return action, epsilon

    def return_q_value(self, observation):
        '''
        Params:
        observation = input value of the state the agent is in

        Returns:
        maximum q value
        '''
        # We will need this function later for plotting the 3D graph

        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        q_values = self.online_network(obs_t.unsqueeze(0))

        return torch.max(q_values).item(), q_values

    def learn(self, batch_size):

        '''
        Params:
        batch_size = number of transitions that will be sampled
        '''

        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(batch_size)

        # Compute targets, note that we use the same neural network to do both! This will be changed later!

        target_q_values = self.target_network(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + self.discount_rate * (1 - dones_t) * max_target_q_values

        # Compute loss

        q_values = self.online_network(observations_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        # Loss, here we take the huber loss!

        loss = F.smooth_l1_loss(action_q_values, targets)
        self.losses.append(loss.item())

        # Uncomment the following code to use the MSE loss instead!
        # loss = F.mse_loss(action_q_values, targets)

        # Gradient descent to update the weights of the neural network
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()

    def update_target_network(self):

        '''
        Complete the method which updates the target network with the parameters of the online network
        '''

        self.target_network.load_state_dict(self.online_network.state_dict())



    def save(self,Path):
        save_dir = Path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.online_network.state_dict(), Path+"Q_network.pt")
        torch.save(self.target_network.state_dict(), Path+"T_network.pt")
        print("Saved successfully")

    def load(self, Path, device):
        self.online_network.load_state_dict(torch.load(Path + "Q_network.pt",map_location=torch.device(device)))
        self.target_network.load_state_dict(torch.load(Path + "T_network.pt",map_location=torch.device(device)))
        print("Loaded successfully")


def training_loop(env, agent, max_epochs, target_=False, batch_size = 1, auto_save = False, cont = None, update_freq = 25, verbose = False):
    '''
    Params:
    env = name of the environment that the agent needs to play
    agent= which agent is used to train
    max_episodes = maximum number of games played
    target = boolean variable indicating if a target network is used (this will be clear later)
    batch_size = amoung of batches it should learn in
    auto_save = int for autosave at each x epoch, set to None to turn off
    cont = List or Tuple for continuing training, [0] is all steps taken, [1] is epoch that it continues from
    update_freq = update frequency of target network in steps

    Returns:
    average_reward_list = a list of averaged rewards over 100 episodes of playing the game
    '''

    obs = env.reset()
    average_reward_list = []
    episode_reward = 0.0
    start_epoch = 0
    all_steps = 0

    print("Training started")
    if cont != None:
        all_steps = cont[0]
        print("Continuing from epoch :", cont[1])
        start_epoch = cont[1]

    for epoch in range(start_epoch, max_epochs):
        done = False
        agent.losses = []
        while not done:
            # print(obs)
            # 4 seems to be the "magic" number of turns that messes up box


            # This works for setting camera angles
            #     env.rob.set_phone_tilt(107.8, 50)
            #     env.rob.set_phone_tilt(108.1, 50)
            #      Angles work ok. Maybe 108.2 better
            #     Maybe punish turning? The more it turns the more its punished

            # counter = 0
            # while counter < 100:
            #     env.rob.set_phone_tilt(107.8, 50)
                # Following code gets an image from the camera
                # image = env.rob.get_image_front()
                # IMPORTANT! `image` returned by the simulator is BGR, not RGB
                # cv2.imwrite("test_pictures_1.png", image)
                # counter += 1
                # env.rob.set_phone_tilt(108.2, 50)
                # Following code gets an image from the camera
                # image = env.rob.get_image_front()
                # IMPORTANT! `image` returned by the simulator is BGR, not RGB
                # cv2.imwrite("test_pictures_2.png", image)
                # counter += 1
                # if counter % 4 == 0 :
                #     env.step(1)
            action, epsilon = agent.choose_action(all_steps, obs)

            new_obs, rew, terminated = env.step(action)
            if verbose :
                # print("obs: ", obs)
                print("rew: ", rew)
                print("got closer?", env.got_closer)
                print("got red?", env.got_red)

            done = terminated
            transition = (obs, action, rew, done, new_obs)
            agent.replay_memory.add_data(transition)
            obs = new_obs

            episode_reward += rew

            # Learn
            agent.learn(batch_size)
            all_steps += 1

            # Update target network
            if target_:
                # Set the target_update_frequency
                target_update_frequency = update_freq
                if all_steps % target_update_frequency == 0:
                    agent.update_target_network()

        # Print some output
        print(20 * '--')
        print('Episode', epoch+1, 'All steps', all_steps)
        print("Avg loss", np.average(agent.losses))
        agent.losses = []
        print('Timesteps', env.time)
        print('Epsilon', epsilon)
        print('Episode Rew', episode_reward)
        agent.replay_memory.add_reward(episode_reward)
        print('Avg Rew', np.mean(agent.replay_memory.reward_buffer))
        print()
        obs = env.reset()
        # Reinitilize the reward to 0.0 after the game is over
        episode_reward = 0.0

        average_reward_list.append(np.mean(agent.replay_memory.reward_buffer))

        if auto_save != None :
            if epoch % auto_save == 0:
                agent.save(f"logs/task3/agent3/backup/autosave_{epoch}/")
                np.savetxt(f"logs/task3/agent3/backup/autosave_{epoch}/all_steps.txt", np.array([all_steps]))



    print("Training done")
    return average_reward_list