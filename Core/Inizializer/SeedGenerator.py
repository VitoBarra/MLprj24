import random
class SeedGenerator:


    def __init__(self,  range_low,range_high, seed ):
        self.range_low = range_low
        self.range_high = range_high
        self.seed = seed
        random.seed(seed)

    def SetSeed(self, seed):
        self.seed = seed
        random.seed(seed)

    def GetSeeds(self , n):
        seedList = [random.randint(self.range_low, self.range_high) for _ in range(n)]
        return seedList

