from Model.Cup.CupModel import TrainCUPModel, GenerateAllPlot_CUP
from Model.Monk.MonkModel import TrainMonkModel, GenerateAllPlot_MONK

if __name__ == '__main__':
    #MONK
    monkNumList = [1,2,3]
    TrainMonkModel(750,50, monkNumList)
    GenerateAllPlot_MONK(monkNumList)

    #CUP
    TrainCUPModel(1000,50)
    GenerateAllPlot_CUP()