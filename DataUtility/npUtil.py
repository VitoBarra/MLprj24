import numpy as np



def arrangeClosed(lower_bound, upper_bound, step):
    return np.arange(lower_bound, upper_bound + step, step)
