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

    def __init__(self, data: np.ndarray, label: np.ndarray, Id: np.ndarray):
        """
        Initializes the DataSet with data and labels.

        :param data: A numpy array containing the input data.
        :param label: A numpy array containing the labels for the data.
        """

        if data is None or label is None:
            raise ValueError("Data and label must be provided.")

        self.Data = DataExamples(data, label, Id)

        self.Training = None
        self.Validation = None
        self.Test = None


    def __iter__(self):
        # Return the instance itself as an iterator
        self.current = 0
        return self


    def __next__(self):
        self.current=self.current+1
        if self.current >= self.Data.DataLength :
            raise StopIteration  # Stop iteration when we've passed the end
        return self.Data.Data[self.current], self.Data.Label[self.current], self.Data.Id[self.current]

    def Split(self, validationPercent: float = 0.15, testPercent: float = 0.1) -> (DataExamples, DataExamples, DataExamples):
        """
        Splits the dataset into training, validation, and test sets.

        :param validationPercent: The fraction of the dataset to be used for validation.
        :param testPercent: The fraction of the dataset to be used for testing.
        :return: The current DataSet object with split data.
        """
        self.Training, self.Validation, self.Test = self.Data.SplitDataset(validationPercent, testPercent)
        self.Data = None
        return self.Training, self.Validation, self.Test


    def Normalize(self, normalizeLable :bool= False,mean: float = None, std: float = None) -> 'DataSet':
        """
        Normalizes the dataset by subtracting the mean and dividing by the standard deviation.

        :param mean: Optional precomputed mean for normalization. Defaults to the dataset mean.
        :param std: Optional precomputed standard deviation for normalization. Defaults to the dataset std.
        :return: The current DataSet object with normalized data.
        :raises ValueError: If normalization is attempted after splitting the dataset.
        """
        if self.Data is None:
            raise ValueError('normalize function must be called before splitDataset')
        self.Data.Normalize(normalizeLable, mean, std)
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

    def PrintData(self) -> None:
        """
        Prints the sizes of the training, validation, and test splits.
        """
        for d, l, id in self:
            print(d,l, id)

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

    def k_fold_cross_validation_split(self, k: int, seed: int = 0) ->[(DataExamples, DataExamples)]:
        """
        Take in input the data we have, and the k fold we want, then it return an array of couple (train_set, test_set)
        :param seed: A dataset, a seed for the shuffle e k for the k-fold
        :return: an array of couple (train_set, test_set) for the k-fold.
        :raises ValueError: If k is too small or too big
        """

        if k <= 0:
            raise ValueError("Fold should be greater than 1")
        if len(self.Data) < k:
            raise ValueError("Fold can't be greater than the number of examples")
        if k == 1:
            return [(self.Data, self.Data)]
        self.Shuffle(seed=seed)


        fold_size = len(self.Data) // k
        remainder = len(self.Data) % k

        folds = []
        start_index = 0
        for i in range(k):
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end_index = start_index + current_fold_size

            fold_data = DataExamples(
                self.Data.Data[start_index:end_index],
                self.Data.Label[start_index:end_index],
                self.Data.Id[start_index:end_index] if self.Data.Id is not None else None
            )
            folds.append(fold_data)
            start_index = end_index

        results: [(DataExamples, DataExamples)] = []
        for i in range(k):
            test_set = folds[i]
            train_set_data = []
            train_set_label = []
            train_set_ids = [] if folds[i].Id is not None else None

            for j, fold in enumerate(folds):
                if j != i:
                    train_set_data.append(fold.Data)
                    train_set_label.append(fold.Label)
                    if fold.Id is not None:
                        train_set_ids.append(fold.Id)

            train_set_data = np.concatenate(train_set_data, axis=0)
            train_set_label = np.concatenate(train_set_label, axis=0)
            train_set_ids = np.concatenate(train_set_ids, axis=0)


            train_set = DataExamples(train_set_data, train_set_label, train_set_ids)
            results.append((train_set, test_set))

        return results

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
