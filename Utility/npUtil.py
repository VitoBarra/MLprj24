import numpy as np



def arrangeClosed(lower_bound, upper_bound, step):
    """
    Create a numpy array of evenly spaced values within a closed interval.

    :param lower_bound: The starting value of the interval.
    :param upper_bound: The ending value of the interval.
    :param step: The spacing between consecutive values in the array.
    :return: A numpy array of values from lower_bound to upper_bound, inclusive.
    """
    arr = np.arange(lower_bound, upper_bound + step, step)
    if arr[-1] > upper_bound:
        arr = arr[:-1]
    return arr

def one_hot_encode(data):
    """
    Perform one-hot encoding on a dataset.

    :param data: A 2D list or numpy array where each column represents a categorical variable.
    :return: A numpy array containing the one-hot encoded representation of the input data.
    """
    # Convert data to numpy array if it's not already
    data = np.array(data)

    # Initialize an empty list to hold one-hot encoded columns
    one_hot_encoded_data = []

    for col in data.T:  # Iterate over columns
        # Find unique values and their indices
        unique_vals, indices = np.unique(col, return_inverse=True)

        # Create one-hot encoding for the column
        one_hot_col = np.eye(len(unique_vals))[indices]

        # Append the one-hot column to the list
        one_hot_encoded_data.append(one_hot_col)

    # Concatenate all one-hot encoded columns horizontally
    return np.hstack(one_hot_encoded_data)