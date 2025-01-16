import numpy as np

from .DataExamples import DataExamples


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
       _Data (DataExamples | None): The complete dataset before splitting.
       Test (DataExamples | None): The test subset of the dataset.
       Validation (DataExamples | None): The validation subset of the dataset.
       Training (DataExamples | None): The training subset of the dataset.
       Kfolds list[tuple[DataExamples,DataExamples]] | None: The number of folds to split the dataset into.
    """
    _Data: DataExamples | None
    Test:       DataExamples | None
    Validation: DataExamples | None
    Training:   DataExamples | None
    Kfolds: list[tuple[DataExamples,DataExamples]] | None

    Splitted: bool

    def __init__(self ):
        self.Kfolds = None
        self._Data = None
        self.Test = None
        self.Validation = None
        self.Training = None
        self.Splitted = False

    @classmethod
    def FromData(cls, data: np.ndarray, label: np.ndarray, Id: np.ndarray):
        """
        Initializes the DataSet with data and labels.

        :param data: A numpy array containing the input data.
        :param label: A numpy array containing the labels for the data.
        :param Id: A numpy array containing the IDs for the data.
        :return: DataSet object.
        """

        if data is None or label is None:
            raise ValueError("DataSet and label must be provided.")
        dataset = DataSet()
        dataset._Data = DataExamples(data, label, Id)
        dataset.DataLength = len(dataset._Data)

        return dataset


    @classmethod
    def FromDataExample(cls, data:DataExamples):
        """
        Initializes the DataSet with data and labels.
        :param data: A DataExamples object.
        :return: A DataSet object.
        """
        dataset = DataSet()
        dataset._Data = data

        return dataset

    @classmethod
    def FromDataExampleTVT(cls, Training:DataExamples, Validation:DataExamples, Test:DataExamples):
        """
        Initializes the DataSet with data and labels from training, validation and test sets.
        :param Training: Training set DataExamples object.
        :param Validation: Validation set DataExamples object.
        :param Test: Test set DataExamples object.
        :return: A DataSet object.
        """

        dataset = DataSet()

        alldata = DataExamples.Clone(Training)
        alldata.Concatenate(Validation)
        alldata.Concatenate(Test)
        dataset._Data = alldata

        dataset.Training = Training
        dataset.Validation = Validation
        dataset.Test = Test



        return dataset

    @classmethod
    def FromDataExampleTV(cls, Training:DataExamples, Test:DataExamples):
        """
        Initializes the DataSet with data and labels from training and test sets.
        :param Training: Training set DataExamples object.
        :param Test: Test set DataExamples object.
        :return: A DataSet object.
        """
        dataset = DataSet()

        alldata = DataExamples.Clone(Training)
        alldata.Concatenate(Test)
        dataset._Data = alldata

        dataset.Training = Training
        dataset.Test = Test

        return dataset

    @classmethod
    def Clone(cls, dataset:'DataSet'):
        """
        Clones the DataSet.
        :param dataset: DataSet object.
        :return: New DataSet object.
        """
        dataset_new = DataSet()
        dataset_new._Data = DataExamples.Clone(dataset._Data)
        dataset_new.Training = DataExamples.Clone(dataset.Training)
        dataset_new.Validation = DataExamples.Clone(dataset.Validation)
        dataset_new.Test = DataExamples.Clone(dataset.Test)

        return dataset_new

    def Split(self, validationPercent: float = 0.15, testPercent: float = 0.1) -> (DataExamples, DataExamples, DataExamples):
        """
        Splits the dataset into training, validation, and test sets.

        :param validationPercent: The fraction of the dataset to be used for validation.
        :param testPercent: The fraction of the dataset to be used for testing.
        :return: The current DataSet object with split data.
        """

        self.Training, self.Validation, self.Test = self._Data.SplitDataset(validationPercent, testPercent)
        self.Splitted = True
        return self.Training, self.Validation, self.Test

    def SplitTV(self, validationPercent: float = 0.15) -> (DataExamples, DataExamples, DataExamples):
        """
        Splits the dataset into training, validation, and test sets.

        :param validationPercent: The fraction of the dataset to be used for validation.
        :return: The current DataSet object with split data.
        """
        self.Training, self.Validation, = self._Data.SplitIn2(validationPercent)
        self.Splitted = True
        return self.Training, self.Validation


    def Standardize(self, Labels :bool= False) -> 'DataSet':
        """
        Standardize the dataset by subtracting the mean and dividing by the standard deviation.

        :param Labels: If true, labels will be normalized.

        :return: The current DataSet object with normalized data.
        :raises ValueError: If normalization is attempted before splitting the dataset.
        """
        if not self.Splitted:
           raise ValueError('Standardize function must be called after splitDataset')

        (statd,statl) = self.Training.Standardize(Labels)
        self.Validation.Standardize(Labels, statd, statl)
        self.Test.Standardize(Labels, statd, statl)
        return self

    def UndoStandardization(self, label :bool= False):
        """
        Undoes the standardization.
        :param label: If true, labels will be unnormalized.
        """
        if self._Data is not None:
            self._Data.Undo_Standardization(label)
        if self.Training is not None:
            self.Training.Undo_Standardization(label)
        if self.Validation is not None:
            self.Validation.Undo_Standardization(label)
        if self.Test is not None:
            self.Test.Undo_Standardization(label)

    def MergeTrainingAndValidation(self):
        """
        Merges the training and validation sets.
        """
        if self.Training is not None and self.Validation is not None:
            self.Training.Concatenate(self.Validation)
            self.Validation = None



    def ToOneHotLabel(self) -> 'DataSet':
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

    def ToOnHotOnData(self):
        """
        Converts data (input) to categorical (one-hot encoded) format for all splits.
        :return:
        """
        if self._Data is not None:
            self._Data.ToCategoricalData()
        if self.Training is not None:
            self.Training.ToCategoricalData()
        if self.Validation is not None:
            self.Validation.ToCategoricalData()
        if self.Test is not None:
            self.Test.ToCategoricalData()
        return self


    def Shuffle(self, seed: int = 0) -> 'DataSet':
        """
        Shuffles the dataset randomly.

        :param seed: An integer seed for reproducibility of shuffling.
        :return: The current DataSet object with shuffled data.
        :raises ValueError: If shuffling is attempted after splitting the dataset.
        """
        if self.Splitted:
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

        if not self.Splitted:
            self._Data.PrintData("all")
        else:
            if self.Training is not None:
                self.Training.PrintData("Training")
            if self.Validation is not None:
                self.Validation.PrintData("Validation")
            if self.Test is not None:
                self.Test.PrintData("Test")




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

    def SetUp_Kfold_TestHoldOut(self, k: int, testRate: float = 0.15) -> None :
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
        if testRate>0:
            data, test_set = self._Data.SplitIn2(testRate)
            self.Test = test_set
        else:
            data = self._Data

        if self.Test is None:
            raise ValueError(f"Test set must be manually set or the testRate must be greater than 0 (it was {testRate} ) ")


        self.Kfolds = self._GenerateKFoldSplit(data, k)

        Total_tr = DataExamples.Clone(self.Kfolds[0][0])
        Total_tr.Concatenate(self.Kfolds[0][1])
        self.Training = Total_tr



    def _GenerateKFoldSplit(self,data: DataExamples , k : int = 5) -> list[tuple[DataExamples, DataExamples]] :
        # Calculate fold sizes
        fold_size = len(data)    // k
        remainder = len(data)    % k

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
        return foldsSplit


    @classmethod
    def init(cls, training: DataExamples, Validation: DataExamples, Test: DataExamples) -> 'DataSet':
        """
        Initializes a DataSet object directly with pre-split training, validation, and test subsets.

        :param training: A DataExamples object for the training subset.
        :param Validation: A DataExamples object for the validation subset.
        :param Test: A DataExamples object for the test subset.
        :return: An instance of the DataSet class.
        """
        instance = DataSet()

        instance.Training = training
        instance.Validation = Validation
        instance.Test = Test
        return instance

    def ApplayTranformationOnLabel(self, param):
        self._Data.Label = param(self._Data.Label)
        if self.Training is not None:
            self.Training.Label = param(self.Training.Label)
        if self.Validation is not None:
            self.Validation.Label = param(self.Validation.Label)
        if self.Test is not None:
            self.Test.Label = param(self.Test.Label)
        return self






