from statistics import mean, variance

from Core.Callback.EarlyStopping import EarlyStopping
from Core.DataSet.DataExamples import DataExamples
from Core.DataSet.DataSet import DataSet
from Core.Metric import Metric
from Core.Tuner.HyperBag import HyperBag
from Core.Initializer.WeightInitializer import GlorotInitializer

from Core.Initializer.SeedGenerator import SeedGenerator


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
    Validates a model based on the hyperparameters and metrics provided.

    Args:
        :param HyperModel_fn (function): The function used to build the model with selected hyperparameters.
        :param best_hpSel (HyperBag): The selected hyperparameters for the model.
        :param NumberOrTrial (int): The number of trials to perform for model validation.
        :param BaselineMetric (Metric): The baseline metric to evaluate model performance.
        :param dataset (DataExamples): The test data to evaluate the model.
        :param training (DataExamples): The training data used for model fitting.
        :param epoch (int, optional): The number of epochs for training (default is 500).
        :param patience (int, optional): The number of epochs with no improvement before stopping (default is 50).
        :param seed (int, optional): The seed value for random number generation (default is 42).

    Returns:
        dict: A dictionary containing the results of the validation, including metrics, hyperparameters, and statistical analysis.
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