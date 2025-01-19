import random
class SeedGenerator:
    """
    A utility class for generating random seeds within a specified range
    and managing a consistent random number generator seed.

    Attributes:
        range_low (int): The lower bound of the range for generated random seeds.
        range_high (int): The upper bound of the range for generated random seeds.
        seed (int): The current seed for the random number generator.
    """

    def __init__(self,  range_low,range_high, seed ):
        self.range_low = range_low
        self.range_high = range_high
        self.seed = seed
        random.seed(seed)

    def SetSeed(self, seed):
        """
        Set the seed for the random number generator

        :param seed: The seed for the random number generator
        """
        self.seed = seed
        random.seed(seed)

    def GetSeeds(self , n):
        """
        Generates a list of random seeds within a specified range.

        :param n: The number of random seeds to generate. Must be a positive integer.
        :return: A list of `n` random integers, each within the range [self.range_low, self.range_high].
        """
        seedList = [random.randint(self.range_low, self.range_high) for _ in range(n)]
        return seedList

