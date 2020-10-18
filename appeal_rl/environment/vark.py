import numpy as np
from .utils import normalize_neg1_pos1


class VARK:
    """
    This class is responsible for calculating the time spent in a certain lesson as well as calculating the VARK reward
    (based on the time spent not based on the difference equation designed previously).
    """

    @staticmethod
    def get_num_differences(current_VARK, preferred_std_gt_VARK):
        return np.sum(np.bitwise_xor(current_VARK, preferred_std_gt_VARK))

    # @staticmethod
    # def calculate_VARK_reward(current_VARK, preferred_std_gt_VARK):
    #     num_differences = VARK.get_num_differences(current_VARK, preferred_std_gt_VARK)
    #     return -0.5 * num_differences + 1

    @staticmethod
    def calculate_VARK_reward(time_spent, params):
        return -1 * normalize_neg1_pos1(time_spent, params['time_spent_Tmin'], params['time_spent_Tmax'])

    @staticmethod
    def calculate_time_spent(current_VARK, preferred_std_gt_VARK, T_min, T_max):
        """
        Direct mapping from the designed excel sheet.
        """
        num_differences = VARK.get_num_differences(current_VARK, preferred_std_gt_VARK)

        if num_differences == 0:
            return np.random.randint(T_min, int(np.ceil(T_min + (T_max - T_min) / 5)))
        if num_differences == 1:
            return np.random.randint(int(np.ceil(T_min + (T_max - T_min) / 5)),
                                     int(np.ceil(T_min + 2 * (T_max - T_min) / 5)))
        if num_differences == 2:
            return np.random.randint(int(np.ceil(T_min + 2 * (T_max - T_min) / 5)),
                                     int(np.ceil(T_min + 3 * (T_max - T_min) / 5)))
        if num_differences == 3:
            return np.random.randint(int(np.ceil(T_min + 3 * (T_max - T_min) / 5)),
                                     int(np.ceil(T_min + 4 * (T_max - T_min) / 5)))
        if num_differences == 4:
            return np.random.randint(int(np.ceil(T_min + 4 * (T_max - T_min) / 5)), T_max)

        raise ValueError("Differences are between 0 and 4")
