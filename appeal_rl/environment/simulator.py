import numpy as np
from environment.constants import Constants
from environment.vark import VARK
from environment.difficulty import Difficulty
from environment.improvement import Improvement
from environment.utils import map_std_level, map_tax_level


class StdSimulatorEnv:
    """
    This is the whole student environment. It follows OpenAI Gym design.
    Assumptions:
    ***********
    1. The interface between the simulator and the RL algorithm works with numerals. However, the insides of the simulator works with classes (E0, E, EM,...)
    2. "Reset" method is used to get a new random student to work with.
    3. "Step" method takes the action from an agent, and returns the found observation, the total reward, and a flag to indicate that the episode has ended.
    """

    def __init__(self, args):
        self.args = args
        self.current_iteration = None
        self.cur_tax_level = None
        self.std_gt_level = None
        self.preferred_std_gt_VARK = None
        self.observation_space = Constants.NUM_TAX * len(Constants.TAX_DIFFICULTIES) * 4 + 1

        self.reset()

    def reset(self):
        """
        Resets the simulator environment.
        - The current iteration becomes zero.
        - Student GT levels are reset to either 'random' or 'specified'.
        - Student Preferred VARK GT is reset to either 'random' or 'specified'.
        - Current taxonomy levels are reset too.
        """
        self.current_iteration = 0
        self.cur_tax_level = np.random.choice(list(Constants.TAX_DIFFICULTIES.values()), size=Constants.NUM_TAX)

        if self.args.preferred_std_gt_VARK_mode == 'specified':
            self.preferred_std_gt_VARK = self.args.preferred_std_gt_VARK
        else:
            self.preferred_std_gt_VARK = np.random.choice([0, 1], size=4)

        if self.args.std_gt_level_mode == 'specified':
            self.std_gt_level = [Constants.STD_LEVELS[elem] for elem in self.args.std_gt_level]
        else:
            self.std_gt_level = np.random.choice(list(Constants.STD_LEVELS.values()), size=Constants.NUM_TAX)

        if self.args.verbose:
            print("Reset successfully")
        return self.step(self.preferred_std_gt_VARK, [0 for _ in range(Constants.NUM_TAX)])

    def step(self, current_VARK, new_tax_level):
        """
        Take a step in the environment. The input is the predicted action from an algorithm.

        :param current_VARK: is a 4-D vector.
        :param new_tax_level: is a 6-D vector that has a value for each taxonomy.

        :return:
        - Observation: a 25-D vector. 4 values per a single taxonomy. We have 6 taxonomies. This gives us 24 values plus a single value representing the time spent in the lesson.

        - Reward: The average reward.

        - Done: A boolean that tells whether the episode is finished or not.


        """
        self.current_iteration += 1

        self.cur_tax_level = Difficulty.change_tax_level(self.cur_tax_level, new_tax_level,
                                                         self.args.delta_difficulties)

        cur_tax_level_mapped = map_tax_level(self.cur_tax_level)
        std_gt_level_mapped = map_std_level(self.std_gt_level)



        # Get observation from the environment, i.e simulating student learning behavior

        observation, stats_reward = Difficulty.sample_stats(cur_tax_level_mapped, std_gt_level_mapped,
                                                            self.current_iteration,
                                                            self.args)

        time_spent = VARK.calculate_time_spent(current_VARK, self.preferred_std_gt_VARK, self.args.time_spent_Tmin,
                                               self.args.time_spent_Tmax)

        observation.extend([time_spent])




        self.std_gt_level = Improvement.improve_std_gt_level(self.cur_tax_level, self.std_gt_level, observation,
                                                             self.current_iteration, self.args)

        # Get the reward from the environment
        vark_reward = VARK.calculate_VARK_reward(time_spent, self.args)

        improvement_speed_reward = Improvement.calculate_speed_reward(cur_tax_level_mapped, std_gt_level_mapped,
                                                                      self.current_iteration, self.args.Tem,
                                                                      self.args.Tmh)
        difficulty_reward = Difficulty.calculate_difficulty_reward(cur_tax_level_mapped, std_gt_level_mapped)

        # Reward factors are set in the JSON config file. Currently, the reward consists of four components:
        # 1. VARK Reward.
        # 2. Improvement Speed Reward.
        # 3. Difficulty Reward.
        # 4. Student Stats Reward.
        reward = self.args.VARK_reward_factor * vark_reward + self.args.improvement_reward_factor * improvement_speed_reward + self.args.difficulty_reward_factor * difficulty_reward + self.args.stats_reward_factor * stats_reward

        # Check that the episode is complete or not
        done = 1
        for i in range(Constants.NUM_TAX):
            if std_gt_level_mapped[i] != 'H1':
                done = 0
                break

        if self.args.verbose:
            print("########################################################")
            print(f"Current Iteration: {self.current_iteration}\n")

            print(f"Current VARK: {current_VARK}")
            print(f"Preferred VARK: {self.preferred_std_gt_VARK}")
            print(f"Difference in VARK: {VARK.get_num_differences(current_VARK, self.preferred_std_gt_VARK)}\n")

            print(f"Current Taxonomy Levels: {cur_tax_level_mapped}")
            print(f"Student GT Levels: {std_gt_level_mapped}\n")

            print(f"Observation: {observation}\n")

            print(f"Stats Reward: {stats_reward}")
            print(f"VARK Reward: {vark_reward}")
            print(f"Improvement Speed Reward: {improvement_speed_reward}")
            print(f"Difficulty Reward: {difficulty_reward}")
            print(f"Average Reward: {reward}\n")

            print(f"Done: {done}\n")
        return observation, reward, done
