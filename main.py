from Model.Cup.CupModel import TrainCUPModel, GenerateAllPlot_CUP
from Model.Monk.MonkModel import GenerateAllPlot_MONK, TrainMonkModel


def ExcMONK():
    #MONK
    monkNumList = [2,4]
    optimizer = [3]
    batch_size = [1]
    TrainMonkModel(150,50, monkNumList, optimizer, batch_size)
    GenerateAllPlot_MONK(monkNumList)

def ExcCUP():
    #CUP
    optimizer = [3]
    batch_size = [-1]
    TrainCUPModel(2,5, optimizer, batch_size )
    GenerateAllPlot_CUP()

if __name__ == '__main__':
    ExcMONK()
    #ExcCUP()

