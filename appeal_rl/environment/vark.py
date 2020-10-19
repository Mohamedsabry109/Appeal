import numpy as np
from .utils import normalize_neg1_pos1


class VARK:
    """
    This class is responsible for calculating the time spent in a certain lesson as well as calculating the VARK reward
    (based on the time spent not based on the difference equation designed previously).
    """

    @staticmethod
    def get_num_differences(current_VARK, preferred_std_gt_VARK):
        """
        This function is used to calculate the number of differences between the predicted VARK and the ground truth VARK.
        
        Parameters
        ----------
        current_VARK : numpy array
            current VARK output by the model.

        preferred_std_gt_VARK : numpy array

            student's ground truth VARK.

        Returns
        -------

        diff : int
            number of differences between the predicted VARK and the ground truth VARK.

        """
        return np.sum(np.bitwise_xor(current_VARK, preferred_std_gt_VARK))

    # @staticmethod
    # def calculate_VARK_reward(current_VARK, preferred_std_gt_VARK):
    #     num_differences = VARK.get_num_differences(current_VARK, preferred_std_gt_VARK)
    #     return -0.5 * num_differences + 1

    @staticmethod
    def calculate_VARK_reward(time_spent, params):
        """
        This function is used to calculate the VARK reward based on the time spent in the lesson, less time means the output is the preffered VARK.
        
        Parameters
        ----------
        time_spent : int
            time spent by the student in a lesson.

        params : dict

            Arguments given in the config file.

        Returns
        -------

        reward : float
            

        """
        return -1 * normalize_neg1_pos1(time_spent, params['time_spent_Tmin'], params['time_spent_Tmax'])



    @staticmethod
    def calculate_time_spent(current_VARK, preferred_std_gt_VARK, T_min, T_max):
        """
        Sampling time spent based on the differences between current VARK and preffered VARK.

        Parameters
        ----------
        current_VARK : numpy array. 
            current VARK predicted by the model.

        preferred_std_gt_VARK : numpy array.

            Preferred VARK by the student.

        T_min : int
            min time to be spent in a lesson
        T_max : int
            max time to be spend in a lesson 

        Returns
        -------

        time : int
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
