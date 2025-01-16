import random
from statistics import mean, variance

from Core.Callback.EarlyStopping import EarlyStopping
from Core.DataSet.DataExamples import DataExamples
from Core.DataSet.DataSet import DataSet
from Core.Metric import Metric
from Core.Tuner.HyperBag import HyperBag
from Core.Inizializer.WeightInitializer import GlorotInitializer

from Core.Inizializer.SeedGenerator import SeedGenerator


def ValidateSelectedModel(
                          HyperModel_fn,
                          best_hpSel: HyperBag,
                          NumberOrTrial: int,
                          MetricsName: list[str],
                          BaselineMetric:Metric,
                          test : DataExamples,
                          training: DataExamples ,
                          ExperimentParam: dict = None,
                          epoch: int = 500,
                          patience: int = 50,
                          seed = 42,

                          ) -> dict :

    totalResult = {"metrics": [], "HP": best_hpSel.hpDic}
    if ExperimentParam is not None:
        totalResult["Test parm"] = ExperimentParam


    res = {key: [] for key in MetricsName}

    tempDataset:DataSet = DataSet()
    tempDataset.Test = test

    seedGenerator = SeedGenerator( 0,1000,seed)

    for i,seed in zip(range(NumberOrTrial), seedGenerator.GetSeeds(NumberOrTrial)):
        training = DataExamples.Clone(training)
        training.Shuffle(seed)
        tempDataset.Training = training
        print(f"Training Model {i + 1}/{NumberOrTrial}...")

        model, optimizer = HyperModel_fn(best_hpSel)
        model.Build(GlorotInitializer(seed))
        model.AddMetric(BaselineMetric)
        callbacks = [EarlyStopping("loss", patience, 0.0001)]
        model.Fit( optimizer,tempDataset, epoch, callbacks)
        totalResult["metrics"].append(model.MetricResults)

        for key, value in model.MetricResults.items():
            res[key].append(value[-1])

        print(f"training model {i + 1} / {NumberOrTrial} " + " | ".join(
            f"{key}:{value[-1]:.4f}" for key, value in res.items()))

    totalResult["MetricStat"] = {key: [mean(value),variance(value)] for key, value in res.items()}
    return  totalResult