from .constants import Constants
import numpy as np
from .utils import map_std_level, map_tax_level, inv_map_std_level
from .utils import normalize_neg1_pos1


class Improvement:
    """
    This class is responsible for improving a simulated student as well as calculating the improvement reward.
    """

    @staticmethod
    def calculate_speed_reward(cur_tax_level_mapped, std_gt_level_mapped, current_iteration, Tem, Tmh):
        """
        This function calculates the total [-1,1] normalized improvement speed reward based on the following (for each taxonomy):
        - current taxonomy level
        - student taxonomy level
        - number of iterations the lesson was repeated
        - Tem, Tmh are hyperparameters found in the designed excel sheet.
        """
        reward = 0
        for i in range(Constants.NUM_TAX):
            difference = Constants.DIFFICULTY_DIFF[(cur_tax_level_mapped[i], std_gt_level_mapped[i])]

            if difference == 0 and cur_tax_level_mapped[i] == 'E':
                if current_iteration <= Tem:
                    reward += 1
                else:
                    reward += (Tem - current_iteration) / current_iteration

            elif difference == 0 and cur_tax_level_mapped[i] == 'M':
                if current_iteration <= Tmh:
                    reward += 1
                else:
                    reward += (Tmh - current_iteration) / current_iteration

        return normalize_neg1_pos1(reward, -Constants.NUM_TAX, Constants.NUM_TAX)

    @staticmethod
    def improve_std_gt_level(cur_tax_level, std_gt_level, observation, current_iteration, params):
        """
        Improve the GT levels of all taxonomies of a student based on the following:
        If a student gets higher grade than a certain threshold, in a time less than a certain threshold,
        with number of hints/attempts lower than a certain threshold;
        then the student improves from a level cluster (E*, M*, H*) to another level cluster (M*, H*, H1).
        """

        def improve_due_to_repetition(cur_level, idx):
            # Repetition allows the student to improve.
            # However, it can't make the level of the student raise completely unless
            # the student gets high grades for example.
            if current_iteration % params.inv_improvement_speed[idx] == 0 and Constants.INV_STD_LEVELS[
                cur_level] not in ['EM',
                                   'MH',
                                   'H', 'H1']:
                cur_level = cur_level + 1
            return cur_level

        std_gt_level_mapped = map_std_level(std_gt_level)

        observation_without_time_spent = np.array(observation[:-1]).reshape(
            (Constants.NUM_TAX, Constants.NUM_TAX_DIFFICULTIES, 4))

        # Even though a student attempts a test with difficulty equals to and greater than his level,
        # to make the student improve we care only about the test of his level.
        for i in range(Constants.NUM_TAX):
            std_gt_level_mapped[i] = Constants.INV_STD_LEVELS[improve_due_to_repetition(std_gt_level[i], i)]
            if std_gt_level_mapped[i][0] == 'E':
                if observation_without_time_spent[i][0][0] <= params.improvement_T_factor_E * params.T_max_E \
                        and observation_without_time_spent[i][0][
                    1] <= params.improvement_hints_factor_E * params.hints_max_E \
                        and observation_without_time_spent[i][0][
                    2] <= params.improvement_atts_factor_E * params.atts_max_E \
                        and observation_without_time_spent[i][0][3] >= params.improvement_G_factor_E * params.G_max_E \
                        and observation_without_time_spent[i][0][0] >= 0:
                    std_gt_level_mapped[i] = 'ME'

            elif std_gt_level_mapped[i][0] == 'M':
                if observation_without_time_spent[i][1][0] <= params.improvement_T_factor_M * params.T_max_M \
                        and observation_without_time_spent[i][1][
                    1] <= params.improvement_hints_factor_M * params.hints_max_M \
                        and observation_without_time_spent[i][1][
                    2] <= params.improvement_atts_factor_M * params.atts_max_M \
                        and observation_without_time_spent[i][1][3] >= params.improvement_G_factor_M * params.G_max_M \
                        and observation_without_time_spent[i][1][0] >= 0:
                    std_gt_level_mapped[i] = 'HM'

            elif std_gt_level_mapped[i][0] == 'H':
                if observation_without_time_spent[i][2][0] <= params.improvement_T_factor_H * params.T_max_H \
                        and observation_without_time_spent[i][2][
                    1] <= params.improvement_hints_factor_H * params.hints_max_H \
                        and observation_without_time_spent[i][2][
                    2] <= params.improvement_atts_factor_H * params.atts_max_H \
                        and observation_without_time_spent[i][2][3] >= params.improvement_G_factor_H * params.G_max_H \
                        and observation_without_time_spent[i][1][0] >= 0:
                    std_gt_level_mapped[i] = 'H1'

        return inv_map_std_level(std_gt_level_mapped)
