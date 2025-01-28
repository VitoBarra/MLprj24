from Utility.npUtil import *

class HyperBag:

    hpDic: dict[str,list[float]]

    def __init__(self):
        """
        Initializes the HyperBag with an empty dictionary for storing hyperparameters.
        """
        self.hpDic = {}

    def __getitem__(self, key:str):
        """
        Retrieves the values associated with a specific hyperparameter.

        :param key: The name of the hyperparameter.
        :return: The list of values for the hyperparameter, or None if the key does not exist.
        """
        if self.hpDic.keys().__contains__(key):
            return self.hpDic[key]
        else:
            return None

    def __setitem__(self, key:str, value:list[float]):
        """
        Sets the values for a specific hyperparameter.

        :param key: The name of the hyperparameter.
        :param value: The list of values to associate with the hyperparameter.
        """
        self.hpDic[key] = value

    def __delitem__(self, key:str):
        """
        Deletes a specific hyperparameter from the dictionary.

        :param key: The name of the hyperparameter to delete.
        """
        del self.hpDic[key]

    def __str__(self):
        """
        Returns a string representation of the hyperparameter dictionary.

        :return: A formatted string of all hyperparameters and their values.
        """
        return self.GetHPString()

    def CheckHP(self, hpName:str) -> None:
        """
        Checks if a hyperparameter is already registered.

        :param hpName: The name of the hyperparameter to check.
        :raises ValueError: If the hyperparameter is already registered.
        """
        if hpName in self.hpDic:
            raise ValueError(f"Hyper parameter '{hpName}' has already bean registered")


    def AddRange(self, hpName:str , lower:float, upper:float, inc:float) -> None:
        """
        Adds a range of values for a hyperparameter.

        :param hpName: The name of the hyperparameter.
        :param lower: The lower bound of the range.
        :param upper: The upper bound of the range.
        :param inc: The increment step for the range.
        :raises ValueError: If the lower bound is greater than the upper bound.
        """
        if lower > upper:
            raise ValueError("Lower bound must be smaller than upper bound")
        self.CheckHP(hpName)
        self.hpDic[hpName] = arrangeClosed(lower, upper, inc).tolist()


    def AddChosen(self, hpName:str, chosen:list[any]) -> None:
        """
        Adds a predefined list of values for a hyperparameter.

        :param hpName: The name of the hyperparameter.
        :param chosen: A list of predefined values.
        :raises ValueError: If the list is empty or contains duplicates.
        """
        if len(chosen) <1:
            raise ValueError("Chosen parameter must have at least length 1")
        if len(chosen) != len(set(chosen)):
            raise ValueError("Chosen parameter cannot contain duplicates")

        self.CheckHP(hpName)
        self.hpDic[hpName] = chosen

    def Keys(self):
        """
        Retrieves the keys of the hyperparameter dictionary.

        :return: A view object of the dictionary's keys.
        """
        return self.hpDic.keys()

    def Values(self):
        """
        Retrieves the values of the hyperparameter dictionary.

        :return: A view object of the dictionary's values.
        """
        return self.hpDic.values()

    def Set(self, hpDic:dict[str,list[float]]):
        """
        Sets the hyperparameter dictionary.

        :param hpDic: A dictionary of hyperparameters and their values.
        :raises ValueError: If the provided dictionary is None.
        """
        if hpDic is None:
            raise ValueError("hpDic cannot be None")
        self.hpDic = hpDic

    def GetHPString(self):
        """
        Generates a formatted string representation of the hyperparameter dictionary.

        :return: A formatted string of hyperparameters and their values.
        """
        return "| "+" | ".join(
            f"{key}:None" if value is None else
            f"{key}:{'True ' if value else 'False'}" if isinstance(value, bool) else
            f"{key}:{value:4}" if isinstance(value, int) else
            f"{key}:{value:.4f}" if isinstance(value, float) else
            f"{key}:{value}"
            for key, value in self.hpDic.items()
        )
