from enum import Enum


class EnumLoss(str,Enum):
    MulticlassnonExlusivCrossentropy = "multiclassnonexlusivcrossentropy"
    BinaryCrossentropy = "binarycrossentropy"
    MSE = "mse"