import numpy as np



def one_hot_encode(data):
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