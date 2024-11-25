import numpy as np
import numpy.random as rng
from numpy import ndarray


class DataExamples(object):
    Label: ndarray
    Data: ndarray
    isCategorical: bool
    DataLength: int

    def __init__(self, data: np.ndarray, label: np.ndarray):
        self.DataLength = data.shape[0]
        if self.DataLength != label.shape[0]:
            raise ValueError('Data and label must have the same length')
        self.Data = data
        self.Label = label
        self.isCategorical = False

    def Concatenate(self, dataLabel):
        if self.isCategorical != dataLabel.isCategorical:
            raise ValueError("each of the dataLabel class must same type of label")
        self.Data = np.concatenate((self.Data, dataLabel.Data), axis=0)
        self.Label = np.concatenate((self.Label, dataLabel.Label), axis=0)

    def Shuffle(self, seed=0):
        rng.seed(seed)
        perm = rng.permutation(self.Data.shape[0])
        self.Data = self.Data[perm,]
        self.Label = self.Label[perm,]

    def SplitDataset(self, validationPercent=0.15, testPercent=0.1):
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
        training = DataExamples(self.Data[:trainingBound], self.Label[:trainingBound])
        validation = DataExamples(self.Data[trainingBound:trainingBound + valBound],
                               self.Label[trainingBound:trainingBound + valBound])
        test = DataExamples(self.Data[trainingBound + valBound:], self.Label[trainingBound + valBound:])
        return training, validation, test

    def SplitIn2(self, rate=0.15):
        if rate <= 0:
            return self, None

        dataLength = self.Data.shape[0]
        splitIndex = int(dataLength * rate)
        dataSplit = DataExamples(self.Data[:splitIndex], self.Label[:splitIndex])
        Data = DataExamples(self.Data[splitIndex:], self.Label[splitIndex:])
        return Data, dataSplit

    def Normalize(self, mean=None, std=None):
        if mean is None:
            mean = np.mean(self.Data, axis=0)
        if std is None:
            std = np.std(self.Data, axis=0)
        self.Data = (self.Data - mean) / std

    def ToCategoricalLabel(self):
        #TODO : tranform to the categorical format
        self.Label = self._ToCategorical(self.Label);
        self.isCategorical = True

    def Slice(self, start, end):
        self.Data = self.Data[start:end]
        self.Label = self.Label[start:end]
        assert self.Data.shape[0] == self.Label.shape[0]

    def FlattenSeriesData(self):
        self.Data = self.Data.reshape(self.Data.shape[0], -1)
        self.Label = self.Label.reshape(self.Label.shape[0], -1)

    def __len__(self):
        return self.DataLength



class DataSet(object):

    Data: DataExamples | None
    Test: DataExamples | None
    Validation: DataExamples | None
    Training: DataExamples | None

    def __init__(self, data: np.ndarray, label: np.ndarray):
        if data is not None and label is not None:
            self.Data = DataExamples(data, label)
        else:
            self.Data = None

        self.Training = None
        self.Validation = None
        self.Test = None

    def SplitDataset(self, validationPercent=0.15, testPercent=0.1):
        self.Training, self.Validation, self.Test = self.Data.SplitDataset(validationPercent, testPercent)
        self.Data = None
        return self

    def Normalize(self, mean=None, std=None):
        if self.Data is None:
            raise ValueError('normalize function must be called before splitDataset')
        self.Data.Normalize(mean, std)
        return self

    def ToCategoricalLabel(self):
        if self.Data is not None:
            self.Data.ToCategoricalLabel()
        if self.Training is not None:
            self.Training.ToCategoricalLabel()
        if self.Validation is not None:
            self.Validation.ToCategoricalLabel()
        if self.Test is not None:
            self.Test.ToCategoricalLabel()
        return self

    def Shuffle(self, seed=0):
        if self.Data is None:
            raise ValueError('shuffle function must be called before splitDataset')
        self.Data.Shuffle(seed)
        return self

    def Unpack(self):
        return self.Training, self.Validation, self.Test

    def PrintSplit(self):
        string = ""
        if self.Training is not None:
            string += f"Training:{self.Training.Data.shape}, "
        if self.Validation is not None:
            string += f"Validation:{self.Validation.Data.shape}, "
        if self.Test is not None:
            string += f"Test:{self.Test.Data.shape}"

        print(string)

    def FlattenSeriesData(self):
        if self.Data is not None:
            self.Data.FlattenSeriesData()
        if self.Training is not None:
            self.Training.FlattenSeriesData()
        if self.Validation is not None:
            self.Validation.FlattenSeriesData()
        if self.Test is not None:
            self.Test.FlattenSeriesData()

    @classmethod
    def init(cls, training, Validation, Test):
        instance = DataSet(None, None)

        instance.Training = training
        instance.Validation = Validation
        instance.Test = Test
        return instance
