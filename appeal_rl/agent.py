import os
import torch
import math
from .dqn.model import Model
from .environment.constants import Constants


class Agent:
    """
    This Agent that will work to deal with student
    """

    def __init__(self, args):
        print("Initializing the Agent")
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        self.checkpoint_dir = args.checkpoint_dir
        self.initial_checkpoint_file = args.initial_checkpoint_file

        self.delta_difficulties = False

        observation_space = Constants.NUM_TAX * len(Constants.TAX_DIFFICULTIES) * 4 + 1
        # 6x3x4 + 1

        self.cur_iteration = 0
        self.start_episode = 0
        # Create a policy network and send it to the GPU.
        self.policy_network = Model(observation_space,
                                    Constants.NUM_TAX,
                                    self.delta_difficulties).to(self.device)

        self.load(self.initial_checkpoint_file)
        self.policy_network = self.policy_network.eval()
        print("Agent initialized successfully")

    def get_policy_action(self, state, network):
        """
        This function gets the current policy action [action with highest Q-Value] in a certain state using the policy network.
        """
        q_vark, q_taxonomies = network.forward(state)

        vark_action = q_vark.max(1)[1]

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
        q_taxonomies_action = [None for _ in range(Constants.NUM_TAX)]
        for i in range(Constants.NUM_TAX):
            q_taxonomies_action[i] = q_taxonomies[i].max(1)[0].detach()

        return q_vark_action, q_taxonomies_action

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

    def load(self, filename="checkpoint.pth.tar"):
        filename = os.path.join(self.checkpoint_dir, filename)
        try:
            print("Loading checkpoint from '{}'".format(filename))
            state = torch.load(filename)
            print("state Loaded")
            self.start_episode = state['start_episode']
            self.cur_iteration = state['cur_iteration']
            print("Network Loading")
            self.policy_network.load_state_dict(state['policy_network'])
            print("Checkpoint Loaded")

        except OSError as _:
            print(f"No checkpoint exists at '{self.checkpoint_dir}'. Skipping...")

    def play(self, state):
        # state, _, _ = self.env.reset() #TODO
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        vark_action, _ = self.get_policy_action(state, self.policy_network)
        vark_action, _ = self.decode_action(vark_action, _)
        return vark_action
