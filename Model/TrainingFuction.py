from statistics import mean, variance

from Core.Callback.EarlyStopping import EarlyStopping
from Core.DataSet.DataExamples import DataExamples
from Core.DataSet.DataSet import DataSet
from Core.Initializer.SeedGenerator import SeedGenerator
from Core.Initializer.WeightInitializer import GlorotInitializer
from Core.Metric import Metric
from Core.Tuner.HyperBag import HyperBag


def AssessmentSelectedModel(
                          HyperModel_fn,
                          best_hpSel: HyperBag,
                          NumberOrTrial: int,
                          BaselineMetric:Metric,
                          data:DataExamples,
                          validation_split: float,
                          test_split: float= None,
                          keep_for_test: float =None,
                          epoch: int = 500,
                          patience: int = 50,
                          seed = 42,
                          ) -> dict :
    """
    Assess the selected hypermodel using the specified hyperparameters through multiple trials.

    :param HyperModel_fn: A callable that builds and returns the model and optimizer given hyperparameters.
    :param best_hpSel: A HyperBag object containing the best hyperparameters for the model.
    :param NumberOrTrial: The number of trials to perform for the assessment.
    :param BaselineMetric: The metric to be used for model evaluation.
    :param data: A DataExamples object representing the dataset.
    :param validation_split: Fraction of the training data to be used for validation.
    :param test_split: Fraction of the dataset to be used for testing (optional, mutually exclusive with keep_for_test).
    :param keep_for_test: Number of samples to keep for testing (optional, mutually exclusive with test_split).
    :param epoch: Number of epochs for training the model.
    :param patience: Patience for early stopping based on validation loss.
    :param seed: Random seed for reproducibility.
    :return: A dictionary containing metrics, hyperparameters, and statistics for the assessment.

    :raises ValueError: If neither `test_split` nor `keep_for_test` is provided.
    """
    if test_split is None and keep_for_test is None:
        raise ValueError("Either test_split or keep_for_test must be provided.")

    totalResult = {"metrics": [], "HP": best_hpSel.hpDic}

    res = {}

    seedGenerator = SeedGenerator( 0,1000,seed)

    for i,seed in zip(range(NumberOrTrial), seedGenerator.GetSeeds(NumberOrTrial)):
        tempData = DataExamples.Clone(data)
        tempData.Shuffle(seed)
        TempDataset: DataSet = DataSet.FromDataExample(tempData)
        if keep_for_test is not None:
            TempDataset.Test = TempDataset.Data[-keep_for_test:]
            TempDataset.Data = TempDataset.Data[:-keep_for_test]
            TempDataset.SplitTV(validation_split)
        else:
            TempDataset.Split(validation_split, test_split)
        print(f"Training Model {i + 1}/{NumberOrTrial}...")

        model, optimizer = HyperModel_fn(best_hpSel)
        model.Build(GlorotInitializer(seed))
        model.AddMetric(BaselineMetric)
        callbacks = [EarlyStopping("val_loss", patience, 0.0001)]
        model.Fit( optimizer,TempDataset, epoch, callbacks)
        totalResult["metrics"].append(model.MetricResults)

        for key, value in model.MetricResults.items():
            if key not in res:
                res[key] = []
            res[key].append(value[-1])

        print(f"training model {i + 1} / {NumberOrTrial} " + " | ".join(
            f"{key}:{value[-1]:.4f}" for key, value in res.items()))

    totalResult["MetricStat"] = {key: [mean(value),variance(value)] for key, value in res.items()}
    return  totalResult