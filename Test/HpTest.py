from Core.Tuner.HyperBag import HyperBag

from Core.Tuner.HpSearch import GridSearch, RandomSearch

if __name__ == '__main__':
    hp = HyperBag()
    hp.AddRange("labda",0.1,0.5,0.05)
    hp.AddChosen("alpha",[0.3,0.4,0.5,0.6])
    hp.AddRange("eta",0.05,0.3,0.05)

    for key in hp.Keys():
        print(f"{key}: {hp[key]}")


    try:
        hp.AddChosen("alpha",[0.3,0.4,0.5,0.6])
    except Exception as exc:
        print(f"\n{exc}\n")


    gs= RandomSearch(hp,10)

    for hyperParam in gs.search():
        print(hyperParam)