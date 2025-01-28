from Model.Cup.CupModel import TrainCUPModel, GenerateAllPlot_CUP
from Model.Monk.MonkModel import TrainMonkModel, GenerateAllPlot_MONK

def ExcMONK():
    #MONK
    monkNumList = [1,2,3,4]
    optimizer = [3]
    batch_size = [32,64]
    TrainMonkModel(250,50, monkNumList, optimizer, batch_size)
    GenerateAllPlot_MONK(monkNumList)

def ExcCUP():
    #CUP
    optimizer = [3]
    batch_size = [-1]
    TrainCUPModel(1000,50, optimizer, batch_size )
    GenerateAllPlot_CUP()

if __name__ == '__main__':
    ExcMONK()
    #ExcCUP()

