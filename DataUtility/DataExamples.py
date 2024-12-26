import numpy as np
from numpy import ndarray
from numpy.random import default_rng


class DataExamples(object):
    """
    Represents a labeled dataset for machine learning tasks.

    This class provides methods for:
      - Concatenating datasets.
      - Shuffling and splitting the dataset.
      - Normalizing the data.
      - Converting labels to categorical format.
      - Flattening the data for models requiring 2D inputs.

    Attributes:
        Label (ndarray): The labels corresponding to the data samples.
        Data (ndarray): The input data samples.
        iD (ndarray): The indices of the data samples.
        DataLength (int): The number of data samples in the dataset.
        isCategorical (bool): Indicates if the labels are in categorical format.
    """
    Label: ndarray
    Data: ndarray
    Id: ndarray
    DataLength: int
    isCategorical: bool

    def __init__(self, data: np.ndarray, label: np.ndarray, Id: np.ndarray) -> None:
        """
        Initializes a DataExamples object with data and labels.

        :param data: A numpy array containing the input data.
        :param label: A numpy array containing the labels for the data.
        :raises ValueError: If the length of data and label do not match.
        """
        self.DataLength = data.shape[0]

        if self.DataLength != label.shape[0]:
            raise ValueError('Data and label must have the same length')
        if self.DataLength != Id.shape[0]:
            raise ValueError('Data and id must have the same length')

        self.Data = data
        self.Label = label
        self.Id = Id
        self.isCategorical = False



    def __iter__(self):
        # Return the instance itself as an iterator
        self.current = 0
        return self


    def __next__(self):
        self.current=self.current+1
        if self.current >= self.DataLength :
            raise StopIteration  # Stop iteration when we've passed the end
        return self.Data[self.current], self.Label[self.current], self.Id[self.current]


    def Concatenate(self, dataExample: 'DataExamples') -> None:
        """
        Concatenates the current dataset with another dataset.

        :param dataExample: A DataExamples object to be concatenated.
        :raises ValueError: If the label types of the datasets do not match.
        """
        if self.isCategorical != dataExample.isCategorical:
            raise ValueError("Each of the dataLabel class must have the same type of label.")
        self.Data = np.concatenate((self.Data, dataExample.Data), axis=0)
        self.Label = np.concatenate((self.Label, dataExample.Label), axis=0)
        self.Id = np.concatenate((self.Id, dataExample.Id), axis=0)

    def Shuffle(self, seed: int = 0) -> None:
        """
        Shuffles the dataset randomly.

        :param seed: An integer seed for reproducibility of shuffling. Default is 0.
        """
        #rng.seed(seed)
        rng = default_rng(seed)
        perm = rng.permutation(self.Data.shape[0])
        self.Data = self.Data[perm,]
        self.Label = self.Label[perm,]
        self.Id = self.Id[perm,]

    def SplitDataset(self, validationPercent: float = 0.15, testPercent: float = 0.1) -> ('DataExamples','DataExamples', 'DataExamples'):
        """
        Splits the dataset into training, validation, and test sets.

        :return: 
        :param validationPercent: The fraction of data to use for validation.
        :param testPercent: The fraction of data to use for testing.
        :return: A tuple (training, validation, test) where each is a DataExamples object.
                 If validationPercent is 0, the second element of the tuple is None.
        :raises ValueError: If the percentages are negative or their sum exceeds 1.
        """
        if validationPercent < 0 or testPercent < 0:
            raise ValueError('Validation and test rate must be in range [0,1]')
        if validationPercent + testPercent > 1:
            raise ValueError('Validation + test rate must be less than 1')
        if validationPercent <= 0:
            training, test = self.SplitIn2(testPercent)
            return training, None, test

        dataLength = self.Data.shape[0]
        trainingBound = int(dataLength * (1 - validationPercent - testPercent))
        valBound = int(dataLength * validationPercent)
        training = DataExamples(self.Data[:trainingBound], self.Label[:trainingBound], self.Id[:trainingBound])
        validation = DataExamples(self.Data[trainingBound:trainingBound + valBound],
                                  self.Label[trainingBound:trainingBound + valBound],
                                  self.Id[trainingBound:trainingBound + valBound])
        test = DataExamples(self.Data[trainingBound + valBound:], self.Label[trainingBound + valBound:],self.Id[trainingBound + valBound:])
        return training, validation, test

    def SplitIn2(self, rate: float = 0.15) -> ('DataExamples','DataExamples'):
        """
        Splits the dataset into two parts.

        :param rate: The fraction of data to include in the first split.
        :return: A tuple (remaining_data, split_data) where each is a DataExamples object.
        """
        if rate <= 0:
            return self, None

        dataLength = self.Data.shape[0]
        splitIndex = int(dataLength * rate)
        dataSplit = DataExamples(self.Data[:splitIndex], self.Label[:splitIndex], self.Id[:splitIndex])
        Data = DataExamples(self.Data[splitIndex:], self.Label[splitIndex:], self.Id[splitIndex:])
        return Data, dataSplit

    def Standardize(self, normalizeLables:bool, datastat: (float, float ) = None, lablestat: (float, float ) = None) -> ((float, float), (float, float)):
        """
        Normalizes the data by subtracting the mean and dividing by the standard deviation.

        :param normalizeLables:  normalize the lables or not.
        :param datastat: An optional precomputed mean for normalization. Defaults to the mean of the dataset.
        :param lablestat: An optional precomputed standard deviation for normalization. Defaults to the std of the dataset.
        """

        (meanD,stdD), self.Data = DataExamples._normalization(self.Data, datastat)
        if normalizeLables:
            (meanL,stdL), self.Label = DataExamples._normalization(self.Label, lablestat)
            return (meanD,stdD), (meanL,stdL)
        return (meanD,stdD), None

    @staticmethod
    def _normalization(data , stat) -> ((float, float), list[float]):
        mean,std = None,None

        if stat is not None:
            mean,std = stat

        if mean is None:
            mean = np.mean(data, axis=0)
        if std is None:
            std = np.std(data, axis=0)
        return (mean,std) , (data - mean) / std

    def ToCategoricalLabel(self) -> None:
        """
        Converts the labels to categorical format (one-hot encoding).

        :raises NotImplementedError: If the method for conversion is not implemented.
        """
        #TODO : transform to the categorical format
        num_classes = np.max(self.Label) + 1
        self.Label = np.eye(num_classes)[self.Label]
        self.Label = self.Label.squeeze()



    def Slice(self, start: int, end: int) -> None:
        """
        Slices the dataset to keep only a specific range of samples.

        :param start: The starting index of the slice.
        :param end: The ending index of the slice.
        :raises AssertionError: If the sliced data and labels have mismatched lengths.
        """
        self.Data = self.Data[start:end]
        self.Label = self.Label[start:end]
        self.Id= self.Id[start:end]
        assert self.Data.shape[0] == self.Label.shape[0] == self.Id.shape[0]

    def FlattenSeriesData(self) -> None:
        """
        Flattens the input data for models requiring 2D inputs.

        The data is reshaped such that each sample is a 1D vector.
        """
        self.Data = self.Data.reshape(self.Data.shape[0], -1)
        self.Label = self.Label.reshape(self.Label.shape[0], -1)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: The number of samples as an integer.
        """
        return self.DataLength

    def _ToCategorical(self, Label: np.ndarray) -> None:
        """
        Generate the data in categorical format.

        """
        pass
