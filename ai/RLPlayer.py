import copy
import math
import random

import numpy as np
import torch

from game.Player import Player


class RLPlayer(Player):
    def __init__(self, name, game_state, action_space, parameters={}):
        super().__init__(name)
        print(f"Player {name} parameters: {parameters}")
        self.action_space = action_space
        self.device = parameters["device"] if "device" in parameters else 'cpu'
        self.map_extracter = MapFeatureExtractor(164, 164, 256).to(self.device)
        observation = self.game_state_to_observation(game_state)
        self.agent = NeuralQLearningAgent(len(observation), action_space, parameters=parameters)

    def action(self, game_state, learning=False):
        observation = self.game_state_to_observation(game_state)
        action = self.agent.get_action(observation, learning)
        return self.action_space[action]

    def process_transition(self, game_state, action, reward, next_game_state, done):
        observation = self.game_state_to_observation(game_state)
        next_observation = self.game_state_to_observation(next_game_state)
        if action == "left":
            action = 0
        elif action == "right":
            action = 2
        else:
            action = 1
        self.agent.process_transition(observation, action, reward, next_observation, done)

    def game_state_to_observation(self, game_state):
        # player_features = ["x", "y", "direction", "speed", "turn_speed", "marker_size", "no_clip"]
        player_features = ["x", "y", "direction"]
        observation = []
        my_player = game_state["players"][self.player_index]

        observation = observation + [float(my_player[feature]) for feature in player_features]
        observation[0] = observation[0] / 500
        observation[1] = observation[1] / 500
        observation[2] = observation[2] / 360

        # min_dist_to_border = np.min([my_player["x"], (500 - my_player["x"]), my_player["y"], (500 - my_player["y"])])
        #
        # observation.append(min_dist_to_border)

        x, y = my_player["x"], my_player["y"]
        radius = 60
        board = game_state["board"]
        features = []
        for angle in [math.radians(my_player["direction"] + (a - 135)) for a in np.linspace(0, 270, 13)]:
            coords = [round(x + radius * math.sin(angle)), round(y + radius * math.cos(angle))]
            interpolated = np.linspace((x, y), coords, 10)
            try:
                is_obstacle = any([board[round(p[0]), round(p[1])] > 0 for p in interpolated[1:]])
            except IndexError:
                is_obstacle = True
            features.append(float(is_obstacle))
        observation += features

        # board = game_state["board"].astype(np.float32)
        # board[board > 0] = 255.0
        # board = resize(board, (164, 164))
        # board[0, :] = 255.0
        # board[-1, :] = 255.0
        # board[:, 0] = 255.0
        # board[:, -1] = 255.0
        #
        # board_tensor = torch.from_numpy(board)
        # board_tensor = board_tensor.view(1, 1, board_tensor.shape[0], board_tensor.shape[1]).to(self.device)
        # map_features = self.map_extracter.forward(board_tensor)
        # observation = observation + map_features.squeeze().tolist()
        return np.array(observation, dtype=np.float32)

    def save_model_weights(self, output_path):
        torch.save(self.agent.q_dash.state_dict(), output_path + "Q")
        torch.save(self.map_extracter.state_dict(), output_path + "me")

    def load_model_weights(self, path, learning=False):
        self.agent.q_dash.load_state_dict(torch.load(path + "Q"))
        self.map_extracter.load_state_dict(torch.load(path + "me"))
        if not learning:
            self.agent.q_dash.eval()
            self.map_extracter.eval()


class MapFeatureExtractor(torch.nn.Module):
    def __init__(self, map_width, map_height, output_features, hidden_count=128):
        super(MapFeatureExtractor, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(4096, output_features)
        )

        # Defining the forward pass

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class ActionApproximation(torch.nn.Module):
    def __init__(self, state_observations_count, action_count, hidden_count=512):
        super(ActionApproximation, self).__init__()
        self.ReLU = torch.nn.ReLU()
        self.dense0 = torch.nn.Linear(state_observations_count, hidden_count)
        self.dense1 = torch.nn.Linear(hidden_count, hidden_count)
        self.dense2 = torch.nn.Linear(hidden_count, action_count)

    def forward(self, x):
        x = x.float()
        x = self.dense0(x)
        x = self.ReLU(x)
        x = self.dense1(x)
        x = self.ReLU(x)
        x = self.dense2(x)
        return x


class Agent:
    def __init__(self):
        pass

    def process_transition(self, observation, action, reward, next_observation, done):
        raise NotImplementedError()

    def get_action(self, observation, learning):
        raise NotImplementedError()


class NeuralQLearningAgent(Agent):
    def __init__(self, observation_len, action_space, parameters={}):
        super().__init__()
        #         torch.manual_seed(42)
        learning_episodes = parameters["learning_episodes"] if "learning_episodes" in parameters else 200

        self.device = parameters["device"] if "device" in parameters else 'cpu'
        self.action_space = action_space
        # PARAMETERS
        self.network_freezing = parameters["network_freezing"] if "network_freezing" in parameters else True
        self.double_q_learning = parameters["double_q_learning"] if "double_q_learning" in parameters else True
        self.batch_learning = parameters["batch_learning"] if "batch_learning" in parameters else True
        self.initial_epsilon = parameters["epsilon"] if "epsilon" in parameters else 0.7
        self.epsilon = self.initial_epsilon
        self.gamma = parameters["gamma"] if "gamma" in parameters else 0.99
        self.learning_rate = parameters["lr"] if "lr" in parameters else 0.001
        self.memory_size = parameters["memory_size"] if "memory_size" in parameters else 10000
        self.memory_start_learning = 1000
        self.batch_size = 128
        self.batch_refresh_interval = 1
        self.network_freezing_i = 3000
        # ...........

        self.q_dash = ActionApproximation(observation_len, len(action_space)).to(self.device)
        if self.network_freezing or self.double_q_learning:
            self.q_dash2 = copy.deepcopy(self.q_dash).to(self.device)

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_dash.parameters(), lr=self.learning_rate)

        self.exploration_weights = [1 / 3, 1 / 3, 1 / 3]
        self.memory = []
        self.memory_index = 0
        self.epsilon_decay_parameter = math.log(
            0.02) / learning_episodes  # (learning_episodes - (learning_episodes // 4))
        self.total_episode_reward = 0
        self.total_reward_memory = []
        self.max_episode_reward = 0
        self.episodes_finished = 0
        self.steps = 0

    def update_approximator(self, batch):
        observation, action, reward, next_observation, done = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[
                                                                                                                  :, 4]
        observation = torch.from_numpy(np.array(observation.tolist())).to(self.device)
        next_observation = torch.from_numpy(np.array(next_observation.tolist())).to(self.device)

        y_pred = self.q_dash.forward(observation)
        action = torch.from_numpy(action[:, np.newaxis].astype(np.int64)).to(self.device)
        score = torch.gather(y_pred, 1, action)
        score = torch.squeeze(score, 1)

        if self.double_q_learning:
            y_n = self.q_dash.forward(next_observation).to(self.device)
            action_n = torch.argmax(y_n, 1, keepdim=True)
            y_next = self.q_dash2.forward(next_observation).to(self.device)
        elif self.network_freezing:
            y_next = self.q_dash2.forward(next_observation).to(self.device)
        else:
            y_next = self.q_dash.forward(next_observation).to(self.device)

        done = done.astype(np.bool_)
        y_next[done] = 0.0

        reward = torch.from_numpy(reward.astype(np.float32)).to(self.device)
        if self.double_q_learning:
            score_next = torch.gather(y_next, 1, action_n)
            score_next = torch.squeeze(score_next, 1)
            target = reward + (self.gamma * score_next)
        else:
            target = reward + (self.gamma * torch.max(y_next, 1).values)
        target = target.float()
        self.optimizer.zero_grad()
        loss = self.loss_function(score, target)
        loss.backward()
        self.optimizer.step()

        if (self.network_freezing or self.double_q_learning) and self.steps % self.network_freezing_i == 0:
            self.q_dash2.load_state_dict(self.q_dash.state_dict())

    def process_transition(self, observation, action, reward, next_observation, done):
        self.steps += 1
        self.total_episode_reward += reward
        if done:
            self.episodes_finished += 1
            self.max_episode_reward = max(self.total_episode_reward, self.max_episode_reward)
            self.total_reward_memory.append(self.total_episode_reward)
            self.total_episode_reward = 0
            if self.epsilon > 0.05:
                self.epsilon = self.initial_epsilon * math.exp(self.episodes_finished * self.epsilon_decay_parameter)

            if self.episodes_finished % 50 == 0:
                print(f"Episode={self.episodes_finished}, epsilon={round(self.epsilon, 4)}, \
                total_steps={self.steps}, max_reward={round(self.max_episode_reward, 4)}, steps_per_episode={round(self.steps / self.episodes_finished, 2)}")

        if self.batch_learning:
            el = (observation, action, reward, next_observation, done)

            if len(self.memory) < self.memory_size:
                self.memory.append(el)
                if len(self.memory) < self.memory_start_learning:
                    return
            else:
                self.memory[self.memory_index] = el
                self.memory_index = (self.memory_index + 1) % self.memory_size

            if self.steps % self.batch_refresh_interval == 0:
                batch = np.array(random.sample(self.memory, self.batch_size), dtype=object)
            else:
                return

        else:
            # One element batch
            batch = np.array((observation, action, reward, next_observation, done), dtype=object)[np.newaxis, :]

        self.update_approximator(batch)

    def get_action(self, observation, learning):
        if learning and random.random() < self.epsilon:
            action = random.choices([0, 1, 2], k=1)[0]
            return action

        observation = torch.from_numpy(observation).to(self.device)
        y_pred = self.q_dash.forward(observation)
        action = torch.argmax(y_pred).item()
        return action
