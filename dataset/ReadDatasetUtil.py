import numpy as np

from Core.DataSet.DataExamples import DataExamples
from Core.DataSet.DataSet import DataSet

def readMonk(file_path:str) -> DataExamples:
    """
    Reads the Monk dataset from a file and returns it as a DataExamples object.

    The dataset contains 8 columns: the first column is the label, the next six are features, and
    the last is an ID.

    :param file_path: Path to the Monk dataset file.
    :return: A DataExamples object containing the dataset (data, labels, and IDs).
    :raises ValueError: If a line doesn't contain exactly 8 elements.
    """
    data = []
    labels = []
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError("we want 8 elements in this file.")

            # The first six values as data array
            data.append([int(x) for x in parts[1:7]])
            # The seventh value as label
            labels.append(int(parts[0]))
            # The eighth value as ID
            ids.append(parts[7])

    # Converts list to NumPy array
    data = np.array(data)
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0], 1)
    ids = np.array(ids)
    de = DataExamples(data, labels, ids)
    return de

def readCUP(file_path:str) -> DataSet:
    """
    Reads the CUP dataset from a file and returns it as a DataSet object.

    The dataset contains 16 columns: the first is the ID, the next 12 are features, and the last three are labels.

    :param file_path: Path to the CUP dataset file.
    :return: A DataSet object containing the dataset (data, labels, and IDs).
    :raises ValueError: If a line doesn't contain exactly 16 elements.
    """
    data = []
    labels = []
    ids = []
    with open(file_path, 'r') as file:
        for _ in range(7):
            next(file)
        for line in file:
            parts = line.strip().split(',')
            if len(parts) != 16:
                raise ValueError("we want 16 elements in this file.")

            # ID (first value)
            ids.append([int(parts[0])])

            # Data (middle values)
            data.append(list(map(float, parts[1:13])))

            # Labels (last 3 values)
            labels.append(list(map(float, parts[13:])))

    # Convert lists to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)
    ids = np.array(ids)
    return DataSet.FromData(data, labels, ids)

def CreateFakeData(nData:int, xdim :int=1, ydim:int=1) -> DataSet:
    """
    Creates a synthetic dataset with random data and binary labels.

    The data is generated using a uniform distribution for the features and random binary labels.

    :param nData: Number of data points to generate.
    :param xdim: Number of input features (default is 1).
    :param ydim: Number of output labels (default is 1).
    :return: A DataSet object containing the synthetic data.
    """
    x = np.random.uniform(0, 1, (nData,xdim))
    y = np.random.choice([0, 1], (nData, ydim))
    id = np.array(range(x.shape[0]))

    return DataSet.FromData(x,y, id)
