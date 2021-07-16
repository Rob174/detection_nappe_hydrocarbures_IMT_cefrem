"""Contains enumerations EnumLoss and EnumDataset"""

from enum import Enum


class EnumLoss(str, Enum):
    MulticlassnonExlusivCrossentropy = "multiclassnonexlusivcrossentropy"
    BinaryCrossentropy = "binarycrossentropy"
    MSE = "mse"


class EnumDataset(str, Enum):
    Train = "tr"
    Valid = "valid"
