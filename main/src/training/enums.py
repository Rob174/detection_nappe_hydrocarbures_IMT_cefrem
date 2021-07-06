from enum import Enum


class EnumLoss(Enum):
    MulticlassnonExlusivCrossentropy = "multiclassnonexlusivcrossentropy"
    BinaryCrossentropy = "binarycrossentropy"
    MSE = "mse"