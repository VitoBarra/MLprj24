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
    _Data: DataExamples | None
    Test:       DataExamples | None
    Validation: DataExamples | None
    Training:   DataExamples | None

    def __init__(self, ):
        self._Data = None
        self.Test = None
        self.Validation = None
        self.Training = None

    @classmethod
    def FromData(cls, data: np.ndarray, label: np.ndarray, Id: np.ndarray):
        """
        Initializes the DataSet with data and labels.

        :param data: A numpy array containing the input data.
        :param label: A numpy array containing the labels for the data.
        """

        if data is None or label is None:
            raise ValueError("Data and label must be provided.")
        dataset = DataSet()
        dataset._Data = DataExamples(data, label, Id)

        dataset.Training = None
        dataset.Validation = None
        dataset.Test = None
        return dataset

    @classmethod
    def FromDataExample(cls, Training:DataExamples, Validation:DataExamples, Test:DataExamples):
        dataset = DataSet()
        dataset.Training = Training
        dataset.Validation = Validation
        dataset.Test = Test
        return dataset

    def Split(self, validationPercent: float = 0.15, testPercent: float = 0.1) -> (DataExamples, DataExamples, DataExamples):
        """
        Splits the dataset into training, validation, and test sets.

        :param validationPercent: The fraction of the dataset to be used for validation.
        :param testPercent: The fraction of the dataset to be used for testing.
        :return: The current DataSet object with split data.
        """
        self.Training, self.Validation, self.Test = self._Data.SplitDataset(validationPercent, testPercent)
        self._Data = None
        return self.Training, self.Validation, self.Test




    def Standardize(self, normalizeLable :bool= False) -> 'DataSet':
        """
        Standardize the dataset by subtracting the mean and dividing by the standard deviation.

        :return: The current DataSet object with normalized data.
        :raises ValueError: If normalization is attempted before splitting the dataset.
        """
        if self._Data is not None:
            raise ValueError('Standardize function must be called after splitDataset')

        (statd,statl) = self.Training.Standardize(normalizeLable)
        self.Validation.Standardize(normalizeLable,statd,statl)
        self.Test.Standardize(normalizeLable,statd,statl)
        return self

    def ToCategoricalLabel(self) -> 'DataSet':
        """
        Converts labels to categorical (one-hot encoded) format for all splits.

        :return: The current DataSet object with categorical labels.
        """
        if self._Data is not None:
            self._Data.ToCategoricalLabel()
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
        if self._Data is None:
            raise ValueError('shuffle function must be called before splitDataset')
        self._Data.Shuffle(seed)
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

    def PrintData(self) -> None:
        """
        Prints the sizes of the training, validation, and test splits.
        """


        if self._Data is not None:
            print("All Data:")
            for d, l, id in self._Data:
                print(d,l, id)

        if self.Training is not None:
            print("Training data:")
            for d, l, id in self.Training:
                print(d, l, id)
        if self.Validation is not None:
            print("Validation data:")
            for d, l, id in self.Validation:
                print(d, l, id)
        if self.Test is not None:
            print("Test data:")
            for d, l, id in self.Test:
                print(d, l, id)

    def FlattenSeriesData(self) -> None:
        """
        Flattens series data for all splits into 2D format where each sample is a vector.
        """
        if self._Data is not None:
            self._Data.FlattenSeriesData()
        if self.Training is not None:
            self.Training.FlattenSeriesData()
        if self.Validation is not None:
            self.Validation.FlattenSeriesData()
        if self.Test is not None:
            self.Test.FlattenSeriesData()

    def Kfold_TestHoldOut(self, k: int, testRate: float = 0.15) -> (DataExamples, [(DataExamples, DataExamples)]):
        """
        Perform k-fold cross-validation, returning a test set and an array of tuples (train_set, val_set).

        :param k: Number of folds (must be >= 2)
        :param testRate: Proportion of data to set aside for the test set
        :return: A tuple containing the test set and a list of k-fold results (train_set, val_set)
        :raises ValueError: If k is less than 2 or greater than the number of remaining examples after test split.
        """
        if k < 2:
            raise ValueError("Number of folds must be at least 2.")
        if len(self._Data) < k:
            raise ValueError("Number of folds cannot exceed the number of examples.")

        # Split into test set and remaining data
        data, test_set = self._Data.SplitIn2(testRate)

        # Calculate fold sizes
        fold_size = len(data)    // k
        remainder = len(data) % k

        # Create folds
        folds = []
        start_index = 0
        for i in range(k):
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end_index = start_index + current_fold_size
            folds.append(DataExamples(
                data.Data[start_index:end_index],
                data.Label[start_index:end_index],
                data.Id[start_index:end_index] if self._Data.Id is not None else None
            ))
            start_index = end_index

        # Generate k-fold cross-validation sets
        foldsSplit = []
        for i in range(k):
            val_set = folds[i]
            train_set_data = []
            train_set_label = []
            train_set_ids = [] if val_set.Id is not None else None

            for j, fold in enumerate(folds):
                if j != i:  # Exclude the validation fold
                    train_set_data.append(fold.Data)
                    train_set_label.append(fold.Label)
                    if fold.Id is not None:
                        train_set_ids.append(fold.Id)

            train_set = DataExamples(
                np.concatenate(train_set_data, axis=0),
                np.concatenate(train_set_label, axis=0),
                np.concatenate(train_set_ids, axis=0) if train_set_ids else None
            )
            foldsSplit.append((train_set, val_set))

        return test_set, foldsSplit


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


