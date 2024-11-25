import numpy as np

from DataUtility.DataExamples import DataExamples


class DataSet(object):
    """
    Manages a complete dataset and its splits for machine learning tasks.

    This class provides functionality to:
       - Split a dataset into training, validation, and test sets.
       - Normalize the dataset.
       - Convert labels to categorical format.
       - Shuffle the dataset.
       - Flatten series data for models requiring 2D inputs.

    Attributes:
       Data (DataExamples | None): The complete dataset before splitting.
       Test (DataExamples | None): The test subset of the dataset.
       Validation (DataExamples | None): The validation subset of the dataset.
       Training (DataExamples | None): The training subset of the dataset.
    """
    Data:       DataExamples | None
    Test:       DataExamples | None
    Validation: DataExamples | None
    Training:   DataExamples | None

    def __init__(self, data: np.ndarray, label: np.ndarray):
        """
        Initializes the DataSet with data and labels.

        :param data: A numpy array containing the input data.
        :param label: A numpy array containing the labels for the data.
        """

        if data is None or label is None:
            raise ValueError("Data and label must be provided.")

        self.Data = DataExamples(data, label)

        self.Training = None
        self.Validation = None
        self.Test = None

    def SplitDataset(self, validationPercent: float = 0.15, testPercent: float = 0.1) -> 'DataSet':
        """
        Splits the dataset into training, validation, and test sets.

        :param validationPercent: The fraction of the dataset to be used for validation.
        :param testPercent: The fraction of the dataset to be used for testing.
        :return: The current DataSet object with split data.
        """
        self.Training, self.Validation, self.Test = self.Data.SplitDataset(validationPercent, testPercent)
        self.Data = None
        return self

    def Normalize(self, mean: float = None, std: float = None) -> 'DataSet':
        """
        Normalizes the dataset by subtracting the mean and dividing by the standard deviation.

        :param mean: Optional precomputed mean for normalization. Defaults to the dataset mean.
        :param std: Optional precomputed standard deviation for normalization. Defaults to the dataset std.
        :return: The current DataSet object with normalized data.
        :raises ValueError: If normalization is attempted after splitting the dataset.
        """
        if self.Data is None:
            raise ValueError('normalize function must be called before splitDataset')
        self.Data.Normalize(mean, std)
        return self

    def ToCategoricalLabel(self) -> 'DataSet':
        """
        Converts labels to categorical (one-hot encoded) format for all splits.

        :return: The current DataSet object with categorical labels.
        """
        if self.Data is not None:
            self.Data.ToCategoricalLabel()
        if self.Training is not None:
            self.Training.ToCategoricalLabel()
        if self.Validation is not None:
            self.Validation.ToCategoricalLabel()
        if self.Test is not None:
            self.Test.ToCategoricalLabel()
        return self

    def Shuffle(self, seed: int = 0) -> 'DataSet':
        """
        Shuffles the dataset randomly.

        :param seed: An integer seed for reproducibility of shuffling.
        :return: The current DataSet object with shuffled data.
        :raises ValueError: If shuffling is attempted after splitting the dataset.
        """
        if self.Data is None:
            raise ValueError('shuffle function must be called before splitDataset')
        self.Data.Shuffle(seed)
        return self

    def Unpack(self) -> (DataExamples,DataExamples,DataExamples):
        """
        Unpacks and returns the training, validation, and test subsets.

        :return: A tuple (training, validation, test) of DataExamples objects.
        """
        return self.Training, self.Validation, self.Test

    def PrintSplit(self) -> None:
        """
        Prints the sizes of the training, validation, and test splits.
        """
        string = ""
        if self.Training is not None:
            string += f"Training:{self.Training.Data.shape}, "
        if self.Validation is not None:
            string += f"Validation:{self.Validation.Data.shape}, "
        if self.Test is not None:
            string += f"Test:{self.Test.Data.shape}"

        print(string)

    def FlattenSeriesData(self) -> None:
        """
        Flattens series data for all splits into 2D format where each sample is a vector.
        """
        if self.Data is not None:
            self.Data.FlattenSeriesData()
        if self.Training is not None:
            self.Training.FlattenSeriesData()
        if self.Validation is not None:
            self.Validation.FlattenSeriesData()
        if self.Test is not None:
            self.Test.FlattenSeriesData()

    @classmethod
    def init(cls, training: DataExamples, Validation: DataExamples, Test: DataExamples) -> 'DataSet':
        """
        Initializes a DataSet object directly with pre-split training, validation, and test subsets.

        :param training: A DataExamples object for the training subset.
        :param Validation: A DataExamples object for the validation subset.
        :param Test: A DataExamples object for the test subset.
        :return: An instance of the DataSet class.
        """
        instance = DataSet(None, None)

        instance.Training = training
        instance.Validation = Validation
        instance.Test = Test
        return instance
