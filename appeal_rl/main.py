from dqn.dqn import DQN
from env.simulator import StdSimulatorEnv
from env.utils import parse_args
from dqn.model import Model
from env.constants import Constants


# To run the program:
# python main.py --config [config_file_path]
# For example: python main.py --config config/conf1.json

def main():
    """
    The main test drive for this project. It constructs the simulator environment. Then, it calls the modified Multi-DQN for training or testing according to the need.
    Note: If you want to test and debug what is happening in the simulator through an episode, make sure that "test" is the chosen mode along with "verbose: true" in the JSON file.
    :return:
    """
    args = parse_args()
    env = StdSimulatorEnv(args)
    dqn = DQN(env, args)

    if args.mode == 'train':
        dqn.train()
    elif args.mode == 'test':
        dqn.test()
    else:
        raise ValueError("Modes are only train and test")


if __name__ == '__main__':
    main()
