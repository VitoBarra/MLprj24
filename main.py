from Model.Cup.CupModel import TrainCUPModel, GenerateAllPlot_CUP, ReadCUP
from Model.Monk.MonkModel import TrainMonkModel, GenerateAllPlot_MONK


def ExcMONK():
    #MONK
    monkNumList = [1,2,3,4]
    optimizer = [3]
    TrainMonkModel(2000,50, monkNumList,optimizer)
    GenerateAllPlot_MONK(monkNumList)

def ExcCUP():
    #CUP
    TrainCUPModel(3000,50)
    GenerateAllPlot_CUP()





if __name__ == '__main__':
    ExcMONK()
    ExcCUP()

