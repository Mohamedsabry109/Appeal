import argparse
import json
import sys
from pprint import pprint

from easydict import EasyDict as edict
from environment.constants import Constants


def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description="Student Learning Simulator")
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('--config', default=None, type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    pprint(config_args)
    print("\n")

    return config_args


def map_std_level(std_gt_level):
    std_gt_level_mapped = [None for _ in range(Constants.NUM_TAX)]
    for i in range(Constants.NUM_TAX):
        std_gt_level_mapped[i] = Constants.INV_STD_LEVELS[std_gt_level[i]]
    return std_gt_level_mapped


def map_tax_level(cur_tax_level):
    cur_tax_level_mapped = [None for _ in range(Constants.NUM_TAX)]
    for i in range(Constants.NUM_TAX):
        cur_tax_level_mapped[i] = Constants.INV_TAX_DIFFICULTIES[cur_tax_level[i]]
    return cur_tax_level_mapped


def inv_map_std_level(std_gt_level_mapped):
    std_gt_level = [None for _ in range(Constants.NUM_TAX)]
    for i in range(Constants.NUM_TAX):
        std_gt_level[i] = Constants.STD_LEVELS[std_gt_level_mapped[i]]
    return std_gt_level


def inv_map_tax_level(cur_tax_level_mapped):
    cur_tax_level = [None for _ in range(Constants.NUM_TAX)]
    for i in range(Constants.NUM_TAX):
        cur_tax_level[i] = Constants.TAX_DIFFICULTIES[cur_tax_level_mapped[i]]
    return cur_tax_level


def normalize_neg1_pos1(value, minimum, maximum):
    # Normalizer function between [-1, 1] based on the min. value and the max. value.
    return 2.0 * (value - minimum) / (maximum - minimum) - 1
