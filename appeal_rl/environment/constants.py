import os


class Constants:
    """
    This class contains the Constants that are used throughout the project.
    """
    def __init__(self):
        pass



    NUM_TAX = 6

    TAX_DIFFICULTIES = {'E': 0, 'M': 1, 'H': 2}
    INV_TAX_DIFFICULTIES = ['E', 'M', 'H']
    NUM_TAX_DIFFICULTIES = len(TAX_DIFFICULTIES)

    STD_LEVELS = {'E0': 0, 'E': 1, 'EM': 2, 'ME': 3, 'M': 4, 'MH': 5, 'H': 6, 'HM': 7, 'H1': 8}
    INV_STD_LEVELS = ['E0', 'E', 'EM', 'ME', 'M', 'MH', 'H', 'HM', 'H1']
    NUM_STD_LEVEL = len(STD_LEVELS)

    DIFFICULTY_DIFF = {
        ('E', 'E0'): 0, ('E', 'E'): 0, ('E', 'EM'): 0, ('E', 'ME'): 1, ('E', 'M'): 2, ('E', 'MH'): 3,
        ('E', 'HM'): 4, ('E', 'H'): 5, ('E', 'H1'): 6,

        ('M', 'E0'): 3, ('M', 'E'): 2, ('M', 'EM'): 1, ('M', 'ME'): 0, ('M', 'M'): 0, ('M', 'MH'): 0,
        ('M', 'HM'): 1, ('M', 'H'): 2, ('M', 'H1'): 3,

        ('H', 'E0'): 6, ('H', 'E'): 5, ('H', 'EM'): 4, ('H', 'ME'): 3, ('H', 'M'): 2, ('H', 'MH'): 1,
        ('H', 'HM'): 0, ('H', 'H'): 0, ('H', 'H1'): 0
    }
