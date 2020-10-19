from .constants import Constants
import numpy as np
from .utils import inv_map_tax_level
from .utils import normalize_neg1_pos1


class Difficulty:
    """
    This is the class responsible for everything related to calculating a new difficulty of a taxonomy along with the rewards.
    """

    @staticmethod
    def calculate_difficulty_reward(cur_tax_level_mapped, std_gt_level_mapped):
        """
        This function calculates the total difficulty reward based on current taxonomies levels and student ground truth levels.
        
        Parameters
        ----------

        param cur_tax_level_mapped : Array
            Array of current taxonomy levels written in letter form. i.e: E,M,H
        param std_gt_level_mapped : Array
            Array of student g.t. taxonomy levels written in letter form. i.e: E,M,H

        Returns
        -------
        a [-1,1] normalized reward based on the designed excel sheet.
        
        """
        reward = 0
        for i in range(Constants.NUM_TAX):
            difference = Constants.DIFFICULTY_DIFF[(cur_tax_level_mapped[i], std_gt_level_mapped[i])]
            if difference == 0:
                reward += 1
            else:
                reward += 0.23 * (1 - difference)
        return normalize_neg1_pos1(reward, -Constants.NUM_TAX, Constants.NUM_TAX)

    @staticmethod
    def sample_exercise_E_stats_per_tax(tax, std_gt, current_iteration, params):
        """
        This function samples the statistics [time, hints, attempts, grade] for a student when his/her tax. level is 'E'.
        Coefficients are selected according to the excel sheet.

        Parameters
        ----------
        tax : dict
            current taxonomy levels predicted by the model

        std_gt : dict 
            current ground truth level of the student

        current_iteration : int
            current iteration per lesson

        params : dict
            Arguments dictionary given by the config files


        Returns
        -------
        a list of statistics [Time, #Hints, #Attempts, Grade] for easy difficulty.

        """
        # Stats are: (Time, #Hints, #Attempts, Grade)
        if (tax, std_gt) == ('E', 'E0') and current_iteration <= params['Tem']:
            t1, t2, t3, t4 = 2, 3, 3, 3
            h1, h2, h3, h4 = 2, 3, 3, 3
            a1, a2, a3, a4 = 2, 3, 3, 3
            g1, g2, g3, g4 = 0, 3, 1, 3

        elif (tax, std_gt) == ('E', 'E') and current_iteration <= params['Tem']:
            t1, t2, t3, t4 = 1, 3, 2, 3
            h1, h2, h3, h4 = 1, 3, 2, 3
            a1, a2, a3, a4 = 1, 3, 2, 3
            g1, g2, g3, g4 = 1, 3, 2, 3

        elif (tax, std_gt) == ('E', 'EM') and current_iteration <= params['Tem']:
            t1, t2, t3, t4 = 0, 3, 1, 3
            h1, h2, h3, h4 = 0, 3, 1, 3
            a1, a2, a3, a4 = 0, 3, 1, 3
            g1, g2, g3, g4 = 2, 3, 3, 3

        elif (tax, std_gt) == ('E', 'E0') and current_iteration > params['Tem']:
            t1, t2, t3, t4 = 2, 3, 3, 3
            h1, h2, h3, h4 = 2, 3, 3, 3
            a1, a2, a3, a4 = 2, 3, 3, 3
            g1, g2, g3, g4 = 0, 3, 1, 3

        elif (tax, std_gt) == ('E', 'E') and current_iteration > params['Tem']:
            t1, t2, t3, t4 = 1, 3, 2, 3
            h1, h2, h3, h4 = 1, 3, 2, 3
            a1, a2, a3, a4 = 1, 3, 2, 3
            g1, g2, g3, g4 = 1, 3, 2, 3

        elif (tax, std_gt) == ('E', 'EM') and current_iteration > params['Tem']:
            t1, t2, t3, t4 = 0, 3, 1, 3
            h1, h2, h3, h4 = 0, 3, 1, 3
            a1, a2, a3, a4 = 0, 3, 1, 3
            g1, g2, g3, g4 = 2, 3, 3, 3

        elif tax == 'E':
            t1, t2, t3, t4 = 0, 3, 1, 3
            h1, h2, h3, h4 = 0, 3, 1, 3
            a1, a2, a3, a4 = 0, 3, 1, 3
            g1, g2, g3, g4 = 2, 3, 3, 3

        else:
            raise ValueError('This method samples only when taxonomy is E')

        # Numpy random functions are used for sampling between minimum and maximum.
        return [np.random.randint(params['T_min_E'] + t1 * (params['T_max_E'] - params['T_min_E']) / t2,
                                  params['T_min_E'] + t3 * (params['T_max_E'] - params['T_min_E']) / t4 + 1),
                np.random.randint(params['hints_min_E'] + h1 * (params['hints_max_E'] - params['hints_min_E']) / h2,
                                  params['hints_min_E'] + h3 * (
                                          params['hints_max_E'] - params['hints_min_E']) / h4 + 1),
                np.random.randint(params['atts_min_E'] + a1 * (params['atts_max_E'] - params['atts_min_E']) / a2,
                                  params['atts_min_E'] + a3 * (params['atts_max_E'] - params['atts_min_E']) / a4 + 1),
                np.random.randint(params['G_min_E'] + g1 * (params['G_max_E'] - params['G_min_E']) / g2,
                                  params['G_min_E'] + g3 * (params['G_max_E'] - params['G_min_E']) / g4 + 1)]

    @staticmethod
    def calculate_stats_reward(T, hints, atts, G, params, type='E'):
        """
        This function calculates the [-1,1] normalized reward based on the statistics for a certain student.
        For example, large time to solve an exam will have a small or negative reward while small time will have a large positive reward.

        Parameters
        ----------
        T : int
            Time taken to finish an exercise.

        hints : int 
            number of hints used in the exercise.

        atts : 
            number of attempts for an exercise.

        G : int
            Grade achieved for an exercise.

        params : dict
            Arguments dictionary given by the config files.
        type : char
            the current tax. level.

        Returns
        -------
        float reward.

        """
        T_reward = -1 * normalize_neg1_pos1(T, params['T_min_' + type], params['T_max_' + type])
        hints_reward = -1 * normalize_neg1_pos1(hints, params['hints_min_' + type], params['hints_max_' + type])
        atts_reward = -1 * normalize_neg1_pos1(atts, params['atts_min_' + type], params['atts_max_' + type])
        G_reward = normalize_neg1_pos1(G, params['G_min_' + type], params['G_max_' + type])
        return 0.25 * T_reward + 0.25 * hints_reward + 0.25 * atts_reward + 0.25 * G_reward

    @staticmethod
    def sample_exercise_M_stats_per_tax(tax, std_gt, current_iteration, params):
        """
        This function samples the statistics [time, hints, attempts, grade] for a student when his/her tax. level is 'M'.
        Coefficients are selected according to the excel sheet.
        
        Parameters
        ----------
        tax : dict
            current taxonomy levels predicted by the model

        std_gt : dict 
            current ground truth level of the student

        current_iteration : int
            current iteration per lesson

        params : dict
            Arguments dictionary given by the config files


        Returns
        -------
        a list of statistics [Time, #Hints, #Attempts, Grade] for easy difficulty.

        """
        # Stats are: (Time, #Hints, #Attempts, Grade)
        if tax == 'E' and std_gt in ['E0', 'E', 'EM', 'ME']:
            t1, t2, t3, t4 = 2, 3, 3, 3
            h1, h2, h3, h4 = 2, 3, 3, 3
            a1, a2, a3, a4 = 2, 3, 3, 3
            g1, g2, g3, g4 = 0, 3, 1, 3

        elif tax == 'E' and std_gt == 'M':
            t1, t2, t3, t4 = 1, 3, 2, 3
            h1, h2, h3, h4 = 1, 3, 2, 3
            a1, a2, a3, a4 = 1, 3, 2, 3
            g1, g2, g3, g4 = 1, 3, 2, 3

        elif tax == 'E' and std_gt in ['MH', 'HM', 'H', 'H1']:
            t1, t2, t3, t4 = 0, 3, 1, 3
            h1, h2, h3, h4 = 0, 3, 1, 3
            a1, a2, a3, a4 = 0, 3, 1, 3
            g1, g2, g3, g4 = 2, 3, 3, 3

        elif tax == 'M' and std_gt in ['E0', 'E', 'EM']:
            t1, t2, t3, t4 = 2, 3, 3, 3
            h1, h2, h3, h4 = 2, 3, 3, 3
            a1, a2, a3, a4 = 2, 3, 3, 3
            g1, g2, g3, g4 = 0, 3, 1, 3

        elif (tax, std_gt) == ('M', 'ME') and current_iteration <= params['Tmh']:
            t1, t2, t3, t4 = 2, 3, 3, 3
            h1, h2, h3, h4 = 2, 3, 3, 3
            a1, a2, a3, a4 = 2, 3, 3, 3
            g1, g2, g3, g4 = 0, 3, 1, 3

        elif (tax, std_gt) == ('M', 'M') and current_iteration <= params['Tmh']:
            t1, t2, t3, t4 = 1, 3, 2, 3
            h1, h2, h3, h4 = 1, 3, 2, 3
            a1, a2, a3, a4 = 1, 3, 2, 3
            g1, g2, g3, g4 = 1, 3, 2, 3

        elif (tax, std_gt) == ('M', 'MH') and current_iteration <= params['Tmh']:
            t1, t2, t3, t4 = 0, 3, 1, 3
            h1, h2, h3, h4 = 0, 3, 1, 3
            a1, a2, a3, a4 = 0, 3, 1, 3
            g1, g2, g3, g4 = 2, 3, 3, 3

        elif (tax, std_gt) == ('M', 'ME') and current_iteration > params['Tmh']:
            t1, t2, t3, t4 = 2, 3, 3, 3
            h1, h2, h3, h4 = 2, 3, 3, 3
            a1, a2, a3, a4 = 2, 3, 3, 3
            g1, g2, g3, g4 = 0, 3, 1, 3

        elif (tax, std_gt) == ('M', 'M') and current_iteration > params['Tmh']:
            t1, t2, t3, t4 = 1, 3, 2, 3
            h1, h2, h3, h4 = 1, 3, 2, 3
            a1, a2, a3, a4 = 1, 3, 2, 3
            g1, g2, g3, g4 = 1, 3, 2, 3

        elif (tax, std_gt) == ('M', 'MH') and current_iteration > params['Tmh']:
            t1, t2, t3, t4 = 0, 3, 1, 3
            h1, h2, h3, h4 = 0, 3, 1, 3
            a1, a2, a3, a4 = 0, 3, 1, 3
            g1, g2, g3, g4 = 2, 3, 3, 3

        elif tax == 'M' and std_gt in ['HM', 'H', 'H1']:
            t1, t2, t3, t4 = 0, 3, 1, 3
            h1, h2, h3, h4 = 0, 3, 1, 3
            a1, a2, a3, a4 = 0, 3, 1, 3
            g1, g2, g3, g4 = 2, 3, 3, 3

        else:
            raise ValueError('This method samples only when taxonomy is E or M')

        # Numpy random functions are used for sampling between minimum and maximum.
        return [np.random.randint(params['T_min_H'] + t1 * (params['T_max_H'] - params['T_min_H']) / t2,
                                  params['T_min_H'] + t3 * (params['T_max_H'] - params['T_min_H']) / t4 + 1),
                np.random.randint(params['hints_min_H'] + h1 * (params['hints_max_H'] - params['hints_min_H']) / h2,
                                  params['hints_min_H'] + h3 * (
                                          params['hints_max_H'] - params['hints_min_H']) / h4 + 1),
                np.random.randint(params['atts_min_H'] + a1 * (params['atts_max_H'] - params['atts_min_H']) / a2,
                                  params['atts_min_H'] + a3 * (params['atts_max_H'] - params['atts_min_H']) / a4 + 1),
                np.random.randint(params['G_min_H'] + g1 * (params['G_max_H'] - params['G_min_H']) / g2,
                                  params['G_min_H'] + g3 * (params['G_max_H'] - params['G_min_H']) / g4 + 1)]

    @staticmethod
    def sample_exercise_H_stats_per_tax(tax, std_gt, current_iteration, params):
        """
        This function samples the statistics [time, hints, attempts, grade] for a student when his/her tax. level is 'H'.
        Coefficients are selected according to the excel sheet.
        
        Parameters
        ----------
        tax : dict
            current taxonomy levels predicted by the model

        std_gt : dict 
            current ground truth level of the student

        current_iteration : int
            current iteration per lesson

        params : dict
            Arguments dictionary given by the config files


        Returns
        -------
        a list of statistics [Time, #Hints, #Attempts, Grade] for easy difficulty.


        """
        # Stats are: (Time, #Hints, #Attempts, Grade)
        if tax == 'E' and std_gt in ['E0', 'E', 'EM', 'ME', 'M', 'MH', 'HM']:
            # Regardless of Tem
            t1, t2, t3, t4 = 2, 3, 3, 3
            h1, h2, h3, h4 = 2, 3, 3, 3
            a1, a2, a3, a4 = 2, 3, 3, 3
            g1, g2, g3, g4 = 0, 3, 1, 3

        elif (tax, std_gt) == ('E', 'H'):
            t1, t2, t3, t4 = 1, 3, 2, 3
            h1, h2, h3, h4 = 1, 3, 2, 3
            a1, a2, a3, a4 = 1, 3, 2, 3
            g1, g2, g3, g4 = 1, 3, 2, 3

        elif (tax, std_gt) == ('E', 'H1'):
            t1, t2, t3, t4 = 0, 3, 1, 3
            h1, h2, h3, h4 = 0, 3, 1, 3
            a1, a2, a3, a4 = 0, 3, 1, 3
            g1, g2, g3, g4 = 2, 3, 3, 3

        elif tax == 'M' and std_gt in ['E0', 'E', 'EM', 'ME', 'M', 'MH', 'HM']:
            # Regardless of Tmh
            t1, t2, t3, t4 = 2, 3, 3, 3
            h1, h2, h3, h4 = 2, 3, 3, 3
            a1, a2, a3, a4 = 2, 3, 3, 3
            g1, g2, g3, g4 = 0, 3, 1, 3

        elif (tax, std_gt) == ('M', 'H'):
            t1, t2, t3, t4 = 1, 3, 2, 3
            h1, h2, h3, h4 = 1, 3, 2, 3
            a1, a2, a3, a4 = 1, 3, 2, 3
            g1, g2, g3, g4 = 1, 3, 2, 3

        elif (tax, std_gt) == ('M', 'H1'):
            t1, t2, t3, t4 = 0, 3, 1, 3
            h1, h2, h3, h4 = 0, 3, 1, 3
            a1, a2, a3, a4 = 0, 3, 1, 3
            g1, g2, g3, g4 = 2, 3, 3, 3

        elif tax == 'H' and std_gt in ['E0', 'E', 'EM', 'ME', 'M', 'MH', 'HM']:
            t1, t2, t3, t4 = 2, 3, 3, 3
            h1, h2, h3, h4 = 2, 3, 3, 3
            a1, a2, a3, a4 = 2, 3, 3, 3
            g1, g2, g3, g4 = 0, 3, 1, 3

        elif (tax, std_gt) == ('H', 'H'):
            t1, t2, t3, t4 = 1, 3, 2, 3
            h1, h2, h3, h4 = 1, 3, 2, 3
            a1, a2, a3, a4 = 1, 3, 2, 3
            g1, g2, g3, g4 = 1, 3, 2, 3

        elif (tax, std_gt) == ('H', 'H1'):
            t1, t2, t3, t4 = 0, 3, 1, 3
            h1, h2, h3, h4 = 0, 3, 1, 3
            a1, a2, a3, a4 = 0, 3, 1, 3
            g1, g2, g3, g4 = 2, 3, 3, 3

        else:
            raise ValueError('This method samples only when taxonomy is E or M or H')

        # Numpy random functions are used for sampling between minimum and maximum.
        return [np.random.randint(params['T_min_M'] + t1 * (params['T_max_M'] - params['T_min_M']) / t2,
                                  params['T_min_M'] + t3 * (params['T_max_M'] - params['T_min_M']) / t4 + 1),
                np.random.randint(params['hints_min_M'] + h1 * (params['hints_max_M'] - params['hints_min_M']) / h2,
                                  params['hints_min_M'] + h3 * (
                                          params['hints_max_M'] - params['hints_min_M']) / h4 + 1),
                np.random.randint(params['atts_min_M'] + a1 * (params['atts_max_M'] - params['atts_min_M']) / a2,
                                  params['atts_min_M'] + a3 * (params['atts_max_M'] - params['atts_min_M']) / a4 + 1),
                np.random.randint(params['G_min_M'] + g1 * (params['G_max_M'] - params['G_min_M']) / g2,
                                  params['G_min_M'] + g3 * (params['G_max_M'] - params['G_min_M']) / g4 + 1)]

    @staticmethod
    def sample_stats(cur_tax_level_mapped, std_gt_level_mapped, current_iteration, params):
        """
        This function samples the statistics [time, hints, attempts, grade] for a student for all taxonomies.
        It also returns the total calculated difficulty reward.
        
        Parameters
        ----------
        cur_tax_level_mapped : dict
            current taxonomy levels predicted by the model

        std_gt_level_mapped : dict 
            current ground truth level of the student

        current_iteration : int
            current iteration per lesson

        params : dict
            Arguments dictionary given by the config files


        Returns
        -------
        stats : nd list 
            all statistics for all taxonomies

        reward achieved in that lesson


        """
        cur_tax_level = inv_map_tax_level(cur_tax_level_mapped)
        stats = []
        reward = 0.0

        for i in range(Constants.NUM_TAX):
            # Min difficulty is zero and max is 2
            # When a student is of level E, the output exercise will be E,M, and H
            # When a student is of level M, the output exercise will be M or H
            # When a student is of level H, the output exercise is only H  
            single_stats = None
            # Maximum reward will be equal to 3
            if cur_tax_level_mapped[i] == 'E':
                single_stats = Difficulty.sample_exercise_E_stats_per_tax(cur_tax_level_mapped[i],
                                                                          std_gt_level_mapped[i],
                                                                          current_iteration, params)
                reward += Difficulty.calculate_stats_reward(*single_stats, params, 'E')

                single_stats.extend(Difficulty.sample_exercise_M_stats_per_tax(cur_tax_level_mapped[i],
                                                                               std_gt_level_mapped[i],
                                                                               current_iteration, params))
                # reward[i] += Difficulty.calculate_stats_reward(*single_stats[4:], params, 'M')

                single_stats.extend(Difficulty.sample_exercise_H_stats_per_tax(cur_tax_level_mapped[i],
                                                                               std_gt_level_mapped[i],
                                                                               current_iteration, params))
                # reward[i] += Difficulty.calculate_stats_reward(*single_stats[8:], params, 'H')

            if cur_tax_level_mapped[i] == 'M':
                single_stats = [-1, -1, -1, -1]
                # reward[i] += len(single_stats) * 0.25

                single_stats.extend(Difficulty.sample_exercise_M_stats_per_tax(cur_tax_level_mapped[i],
                                                                               std_gt_level_mapped[i],
                                                                               current_iteration, params))
                reward += Difficulty.calculate_stats_reward(*single_stats[4:], params, 'M')

                single_stats.extend(Difficulty.sample_exercise_H_stats_per_tax(cur_tax_level_mapped[i],
                                                                               std_gt_level_mapped[i],
                                                                               current_iteration, params))
                # reward[i] += Difficulty.calculate_stats_reward(*single_stats[8:], params, 'H')

            if cur_tax_level_mapped[i] == 'H':
                single_stats = [-1, -1, -1, -1]
                # reward[i] += len(single_stats) * 0.25

                single_stats.extend([-1, -1, -1, -1])
                # reward[i] += len(single_stats) * 0.25

                single_stats.extend(Difficulty.sample_exercise_H_stats_per_tax(cur_tax_level_mapped[i],
                                                                               std_gt_level_mapped[i],
                                                                               current_iteration, params))
                reward += Difficulty.calculate_stats_reward(*single_stats[8:], params, 'H')

            stats.extend(single_stats)

        # Returned dimension is 6x3x4. ie, 6 taxonomies, 3 exercises, with 4 values
        return stats, normalize_neg1_pos1(reward, -1.0 * Constants.NUM_TAX, 1.0 * Constants.NUM_TAX)

    @staticmethod
    def change_tax_level(cur_tax_level, new_tax_level, delta_difficulties):
        """
        This method changes the current taxonomy level by a delta vector of taxonomies.
        Note that: the minimum taxonomy is 'E' and the maximum taxonomy is 'H'.

        Parameters
        ----------
        cur_tax_level : dict
            current taxonomy
        new_tax_level : dict
            new taxonomy if delta_difficulties are used

        delta_difficulties : boolean
            True is we uses delta_difficulties

        Returns
        -------

        new_tax_level : dict
            new taxonomy if delta_difficulties are used
        """
        if delta_difficulties:
            new_tax_level = np.add(cur_tax_level, new_tax_level)
            new_tax_level = np.maximum(Constants.TAX_DIFFICULTIES['E'], new_tax_level)
            new_tax_level = np.minimum(Constants.TAX_DIFFICULTIES['H'], new_tax_level)
        return new_tax_level
