from numpy import ndarray
from numpy.random import default_rng
from Utility.npUtil import *


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
        Id (ndarray): The indices of the data samples.
        isCategorical (bool): Indicates if the labels are in categorical format.
    """
    Data: ndarray | None
    Label: ndarray | None
    Id: ndarray | None
    isCategorical: bool
    LabelStat: (float,float)
    DataStat: (float,float)

    def __init__(self) -> None:
        self.DataStat = None
        self.Data = None
        self.Label = None
        self.Id = None
        self.LabelStat = None
        self.isCategorical = False



    @classmethod
    def FromData(cls,data: np.ndarray, label: np.ndarray = None, Id: np.ndarray = None):
        """
        Initializes a DataExamples object with data and labels.

        :param data: A numpy array containing the input data.
        :param label: A numpy array containing the labels for the data.
        :param Id:  A numpy array containing the ids for the data
        :raises ValueError: If the length of data and label do not match.
        """
        dataExample = cls()
        if label is not None and data.shape[0] != label.shape[0]:
            raise ValueError('DataSet and label must have the same length')
        if Id is not None and data.shape[0] != Id.shape[0]:
            raise ValueError('DataSet and id must have the same length')

        dataExample.Data = data
        dataExample.Label = label
        dataExample.Id = Id
        return dataExample

    @classmethod
    def Clone(cls, dataExamples: 'DataExamples'):
        """
        Clones a DataExamples object.

        :param dataExamples: A DataExamples object.
        :return: A new DataExamples object.
        """
        if dataExamples is None:
            return None
        return DataExamples.FromData(dataExamples.Data, dataExamples.Label, dataExamples.Id)



    def __iter__(self):
        # Return the instance itself as an iterator
        self.current = 0
        return self


    def __next__(self):
        self.current=self.current+1
        if self.current >= len(self.Data) :
            raise StopIteration  # Stop iteration when we've passed the end
        return self.Data[self.current], self.Label[self.current], self.Id[self.current]

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: The number of samples as an integer.
        """
        return self.Data.shape[0]


    def Concatenate(self, dataExample: 'DataExamples') -> None:
        """
        Concatenates the current dataset with another dataset.

        :param dataExample: A DataExamples object to be concatenated.
        :raises ValueError: If the label types of the datasets do not match.
        """
        if self.isCategorical != dataExample.isCategorical:
            raise ValueError("Each of the dataLabel class must have the same type of label.")

        self.Data = np.concatenate((self.Data, dataExample.Data), axis=0) if self.Data is not None else dataExample.Data
        self.Label = np.concatenate((self.Label, dataExample.Label), axis=0) if self.Label is not None else dataExample.Label
        self.Id = np.concatenate((self.Id, dataExample.Id), axis=0) if self.Id is not None else dataExample.Id

    def Shuffle(self, seed: int = None) -> None:
        """
        Shuffles the dataset randomly.

        :param seed: An integer seed for reproducibility of shuffling. Default is 0.
        """

        rng = default_rng(seed)
        perm = rng.permutation(self.Data.shape[0])
        self.Data = self.Data[perm,]
        self.Label = self.Label[perm,]
        self.Id = self.Id[perm,]

    def SplitDataset(self, validationPercent: float = 0.15, testPercent: float = 0.1) -> ('DataExamples','DataExamples', 'DataExamples'):
        """
        Splits the dataset into training, validation, and test sets.

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
        training = DataExamples.FromData(self.Data[:trainingBound], self.Label[:trainingBound], self.Id[:trainingBound])
        validation = DataExamples.FromData(self.Data[trainingBound:trainingBound + valBound],
                                  self.Label[trainingBound:trainingBound + valBound],
                                  self.Id[trainingBound:trainingBound + valBound])
        test = DataExamples.FromData(self.Data[trainingBound + valBound:], self.Label[trainingBound + valBound:],self.Id[trainingBound + valBound:])
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
        dataSplit = DataExamples.FromData(self.Data[:splitIndex], self.Label[:splitIndex], self.Id[:splitIndex])
        Data = DataExamples.FromData(self.Data[splitIndex:], self.Label[splitIndex:], self.Id[splitIndex:])
        return Data, dataSplit

    def Standardize(self, standardizeLabel:bool, datastat: (float, float) = None, labelstat: (float, float) = None) -> ((float, float), (float, float)):
        """
        Normalizes the data by subtracting the mean and dividing by the standard deviation.

        :param standardizeLabel:  normalize the labels or not.
        :param datastat: An optional precomputed mean for normalization. Defaults to the mean of the dataset.
        :param labelstat: An optional precomputed standard deviation for normalization. Defaults to the std of the dataset.
        :return: A tuple (mean, std)
        """

        self.DataStat, self.Data = DataExamples.Standardization(self.Data, datastat)
        if standardizeLabel:
            self.LabelStat, self.Label = DataExamples.Standardization(self.Label, labelstat)

        return self.DataStat, self.LabelStat

    def Undo_Standardization(self, normalizeLabels: bool) -> None:
        """
        Reverses the standardization of the data and optionally the labels.

        :param normalizeLabels: Indicates whether to reverse standardization for the labels.
        """

        if self.DataStat is not None:
            meanD, stdD = self.DataStat
            if stdD != 0:  # Avoid division by zero
                self.Data = (self.Data * stdD) + meanD

        if normalizeLabels and  self.LabelStat is not None:
            meanL, stdL = self.LabelStat
            if stdL != 0:  # Avoid division by zero
                self.Label = (self.Label * stdL) + meanL

    def Undo_Standardization_ExternalData(self, Data, labels) :
        """
        Reverses the standardization of the data and the labels.

        :param Data: Data to be reversed.
        :param labels: Labels to be reversed.
        """

        if Data is None:
            raise ValueError("DataSet must be provided.")

        resultData,resultLabels = None,None
        if self.DataStat is not None:
            meanD, stdD = self.DataStat
            resultData = (self.Data * stdD) + meanD

        if labels is not None and  self.LabelStat is not None:
            meanL, stdL = self.LabelStat
            resultLabels = (self.Label * stdL) + meanL

        return resultData, resultLabels


    @staticmethod
    def Standardization(data, stat) -> ((float, float), list[float]):
        """
         Standardizes the given data by subtracting the mean and dividing by the standard deviation.

        :param data: A np array containing the data to be standardized. Each element is expected to be a numeric value.
        :param stat: A tuple (mean, std) containing the precomputed mean and standard deviation.
                     If None, the function computes these values from the `data`.

        :return: A tuple containing:
                 1. A tuple (mean, std) where `mean` is the computed or provided mean,
                    and `std` is the computed or provided standard deviation.
                 2. A list or array of standardized values, calculated as (data - mean) / std.
        """
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
        """
        self.Label = one_hot_encode(self.Label)

    def ToCategoricalData(self):
        """
        Converts the labels to categorical format (one-hot encoding).
        """
        self.Data = one_hot_encode(self.Data)




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




    def PrintData(self, name :str):
        """
        Prints the data to the console.
        :param name: The name of the dataset.
        """
        print(f"{name} data {len(self.Data)} :")
        for d, l, id in self:
            print(d, l, id)

    def CutData(self,  finish,start=0):
        """
        Take only a certain part of samples from the dataset.

        :param finish: Finish index for the cut.
        :param start: Start position of the cut.
        """

        self.Data = self.Data[start:finish]
        self.Label = self.Label[start:finish]
        if self.Id is not None:
            self.Id = self.Id[start:finish]




