from tensorboardX import SummaryWriter
import torch
import math
import torch.optim as optim
import torch.nn.functional as F
from .replay_memory import ReplayMemory
from ..environment.constants import Constants
from .model import Model
import random
import os



class DQN:
    """
    The core of the reinforcement learning is our modified Deep Q-Network with multiple heads (in our case 7 heads);
    one for each taxonomy level (6 levels) and one for VARK prediction. Total loss is the combined weighted loss from each head.
    This is a direct implementation of a multi-task learning architecture with a shared encoder.
    """

    def __init__(self, env, args):
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.env = env
        self.batch_size = args.batch_size
        self.discount_factor = args.discount_factor
        self.checkpoint_dir = args.checkpoint_dir
        self.num_episodes = args.num_episodes
        self.delta_difficulties = args.delta_difficulties
        self.save_every_iterations = args.save_every_iterations
        self.print_every_iterations = args.print_every_iterations
        self.plot_every_iterations = args.plot_every_iterations
        self.target_update_every = args.target_update_every
        self.learning_rate = args.learning_rate
        self.max_unroll_depth = args.max_unroll_depth
        self.initial_checkpoint_file = args.initial_checkpoint_file
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay

        self.cur_iteration = 0
        self.start_episode = 0
        self.writer = SummaryWriter(os.path.join("experiments", self.checkpoint_dir))
        # Create a policy network and send it to the GPU.
        self.policy_network = Model(env.observation_space, Constants.NUM_TAX, self.delta_difficulties).to(self.device)

        if args.mode == 'train':
            # RMSprop is the commonly used optimization algorithm.
            self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=self.learning_rate)

            # DQN is mostly used with ReplayMemory as it has proven an increase in performance;
            # due to training set diversity.
            self.replay_memory = ReplayMemory(args.memory_size)

            # Resume training if it was stopped.
            self.load(self.initial_checkpoint_file)

            # Create a target network and send it to the GPU in the training phase ONLY.
            self.target_network = Model(env.observation_space, Constants.NUM_TAX, self.delta_difficulties).to(
                self.device)
            # Resume training with the same parameters of the policy network.
            self.target_network.load_state_dict(self.policy_network.state_dict())

            # It's used as an evaluation network; so put it in the "eval" mode.
            self.target_network.eval()

        if args.mode == 'test':
            self.load(self.initial_checkpoint_file, mode=args.mode)
            self.policy_network = self.policy_network.eval()

    def get_policy_action(self, state, network):
        """
        This function gets the current policy action [action with highest Q-Value] in a certain state using the policy network.
        """
        q_vark, q_taxonomies = network.forward(state)

        vark_action = q_vark.max(1)[1]
        # .max(1)[1] gets the index of the max value
        taxonomies_action = [None for _ in range(Constants.NUM_TAX)]
        for i in range(Constants.NUM_TAX):
            taxonomies_action[i] = q_taxonomies[i].max(1)[1]
        return vark_action, taxonomies_action

    def get_action_value(self, state, vark_action, taxonomies_action, network):
        """
        This function gets the current policy action value in a certain state using the policy network.
        """
        q_vark, q_taxonomies = network.forward(state)

        q_vark_action = torch.squeeze(torch.gather(q_vark, 1, vark_action.unsqueeze(1)), 1)
        q_taxonomies_action = [None for _ in range(Constants.NUM_TAX)]
        for i in range(Constants.NUM_TAX):
            q_taxonomies_action[i] = torch.squeeze(torch.gather(q_taxonomies[i], 1, taxonomies_action[i].unsqueeze(1)),
                                                   1)

        return q_vark_action, q_taxonomies_action

    def get_max_action_value(self, state, network):
        """
        This function gets the maximum policy action value in a certain state using the policy network.
        """
        q_vark, q_taxonomies = network.forward(state)

        q_vark_action = q_vark.max(1)[0].detach()
        # .max(1)[0] gets the max value
        q_taxonomies_action = [None for _ in range(Constants.NUM_TAX)]
        for i in range(Constants.NUM_TAX):
            q_taxonomies_action[i] = q_taxonomies[i].max(1)[0].detach()

        return q_vark_action, q_taxonomies_action

    def get_ep_greedy_action(self, state, network):
        """
        This function uses a greedy policy to sample an action.
        It either gets the action with the maximum Q-value or selects a random action with probability (epsilon)
        """
        sample = random.random()
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.cur_iteration / self.eps_decay)

        if sample > eps:
            return self.get_policy_action(state, network)
        else:
            vark_action = torch.randint(0, 16, (state.shape[0],), device=self.device, dtype=torch.int64)

            taxonomies_action = [None for _ in range(Constants.NUM_TAX)]
            for i in range(Constants.NUM_TAX):
                if self.delta_difficulties:
                    taxonomies_action[i] = torch.randint(0, 5, (state.shape[0],), device=self.device, dtype=torch.int64)
                else:
                    taxonomies_action[i] = torch.randint(0, 3, (state.shape[0],), device=self.device, dtype=torch.int64)

            return vark_action, taxonomies_action

    def decode_action(self, vark_action, taxonomies_action):
        """
        This function decodes the action taken. For example: it converts a VARK of 4 to a VARK vector of [0100].
        """
        decoded_vark_action = [int(x) for x in '{:04b}'.format(int(vark_action.item()))]

        decoded_taxonomies_action = [None for _ in range(Constants.NUM_TAX)]
        for i in range(Constants.NUM_TAX):
            decoded_taxonomies_action[i] = int(taxonomies_action[i][0].item())
            if self.delta_difficulties:
                decoded_taxonomies_action[i] = int(taxonomies_action[i][0].item()) - 2

        return decoded_vark_action, decoded_taxonomies_action

    def train_single_batch(self):
        """
        This function trains the policy network using a single batch sampled from the replay memory.
        """
        if self.replay_memory.size < self.batch_size:
            return None

        self.cur_iteration += 1

        states, vark_actions, taxonomy_actions, rewards, next_states, dones = zip(
            *self.replay_memory.sample(self.batch_size))

        state_batch = torch.cat(states)
        reward_batch = torch.cat(rewards)
        vark_action_batch = torch.cat(vark_actions)
        next_state_batch = torch.cat(next_states)
        done_batch = torch.cat(dones)

        taxonomy_actions = list(zip(*taxonomy_actions))
        taxonomy_action_batch = [None for _ in range(Constants.NUM_TAX)]
        for i in range(Constants.NUM_TAX):
            taxonomy_action_batch[i] = torch.cat(taxonomy_actions[i])

        state_vark_action_value, state_taxonomies_action_value = self.get_action_value(state_batch,
                                                                                       vark_action_batch,
                                                                                       taxonomy_action_batch,
                                                                                       self.policy_network)

        pred_next_state_vark_action_value, pred_next_state_taxonomies_action_value = self.get_max_action_value(
            next_state_batch, self.target_network)

        next_state_vark_action_value = torch.zeros(self.batch_size, device=self.device)
        next_state_vark_action_value[done_batch == 0] = pred_next_state_vark_action_value[done_batch == 0]

        expected_state_vark_action_value = reward_batch + self.discount_factor * next_state_vark_action_value
        expected_state_taxonomies_action_value = [None for _ in range(Constants.NUM_TAX)]
        for i in range(Constants.NUM_TAX):
            next_state_taxonomies_action_value = torch.zeros(self.batch_size, device=self.device)
            next_state_taxonomies_action_value[done_batch == 0] = pred_next_state_taxonomies_action_value[i][
                done_batch == 0]
            expected_state_taxonomies_action_value[
                i] = reward_batch + self.discount_factor * next_state_taxonomies_action_value

        # Loss is computed on the fly here using PyTorch.
        vark_loss = F.smooth_l1_loss(state_vark_action_value, expected_state_vark_action_value)
        taxonomy_loss = 0
        for i in range(Constants.NUM_TAX):
            taxonomy_loss += F.smooth_l1_loss(state_taxonomies_action_value[i],
                                              expected_state_taxonomies_action_value[i])
        loss = vark_loss + taxonomy_loss

        # Backpropagation is done here.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return vark_loss, taxonomy_loss

    def save(self, filename="checkpoint.pth.tar"):
        state = {
            'start_episode': self.start_episode,
            'cur_iteration': self.cur_iteration + 1,
            'policy_network': self.policy_network.state_dict(),
            'optimizer_params': self.optimizer.state_dict()
        }
        # Save the state
        torch.save(state, os.path.join("experiments", self.checkpoint_dir, filename))

    def load(self, filename="checkpoint.pth.tar", mode='train'):
        filename = os.path.join("experiments", self.checkpoint_dir, filename)
        try:
            print("Loading checkpoint from '{}'".format(filename))
            state = torch.load(filename)

            self.start_episode = state['start_episode']
            self.cur_iteration = state['cur_iteration']
            self.policy_network.load_state_dict(state['policy_network'])

            if mode == 'train':
                self.optimizer.load_state_dict(state['optimizer_params'])

            print(f"Checkpoint loaded successfully from {filename} at iteration {self.cur_iteration}\n")
        except OSError as _:
            print(f"No checkpoint exists at '{self.checkpoint_dir}'. Skipping...")
            print("First time to train...\n")

    def train(self):
        """
        This function trains the DQN network for a certain number of episodes with a max. unrolling depth.
        It also computes the statistics of the training process and sends them to Tensorboard for visualization purposes.

        Here is the scenario: It takes a step in the environemnt using eps. greedy policy.
        Then, it adds the obs, rewards, etc. to the replay memory.
        Then, it samples from the replay memory a batch of a certain size.
        Finally, it trains the policy network.
        Note that: after each number of steps, policy network parameters are copied to the target network.
        """
        for episode in range(self.start_episode, self.num_episodes):
            state, _, _ = self.env.reset()
            running_reward_mean = None
            state = torch.tensor([state], device=self.device, dtype=torch.float32)

            for episodic_iterations in range(self.max_unroll_depth):

                self.policy_network = self.policy_network.eval()
                vark_action, taxonomies_action = self.get_ep_greedy_action(state, self.policy_network)
                next_state, reward, done = self.env.step(*self.decode_action(vark_action[0], taxonomies_action))

                if not running_reward_mean:
                    running_reward_mean = reward
                else:
                    running_reward_mean = 0.3 * running_reward_mean + 0.7 * reward

                next_state = torch.tensor([next_state], device=self.device, dtype=torch.float32)
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                done = torch.tensor([done], device=self.device)
                self.replay_memory.add(state, vark_action, taxonomies_action, reward, next_state, done)
                state = next_state

                self.policy_network = self.policy_network.train()

                losses = self.train_single_batch()

                if not losses:
                    continue

                vark_loss, taxonomy_loss = losses

                if self.cur_iteration % self.save_every_iterations == 0:
                    self.save(self.initial_checkpoint_file)

                if self.cur_iteration % self.print_every_iterations == 1:
                    print(
                        f"Training @ iteration: {self.cur_iteration}, vark_loss: {vark_loss}, taxonomy_loss: {taxonomy_loss}, running_reward_mean: {running_reward_mean}")

                if self.cur_iteration % self.plot_every_iterations == 0:
                    self.writer.add_scalar('VARK_Loss', vark_loss, self.cur_iteration)
                    self.writer.add_scalar('Taxonomy_Loss', taxonomy_loss, self.cur_iteration)
                    self.writer.add_scalar('Total_Loss', vark_loss + taxonomy_loss, self.cur_iteration)
                    self.writer.add_scalar('Running_Reward_Mean', running_reward_mean, self.cur_iteration)

                if self.cur_iteration % self.target_update_every == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())

                if done[0].item() == 1:
                    # print(f"Episode {episode} terminated!")
                    self.writer.add_scalar('Done_after_iterations', episodic_iterations, self.cur_iteration)
                    break

    def test(self):
        state, _, _ = self.env.reset()
        state = torch.tensor([state], device=self.device, dtype=torch.float32)

        for episodic_iterations in range(self.max_unroll_depth):

            vark_action, taxonomies_action = self.get_policy_action(state, self.policy_network)
            next_state, reward, done = self.env.step(*self.decode_action(vark_action[0], taxonomies_action))

            next_state = torch.tensor([next_state], device=self.device, dtype=torch.float32)
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
            done = torch.tensor([done], device=self.device)
            state = next_state

            if done[0].item() == 1:
                break
