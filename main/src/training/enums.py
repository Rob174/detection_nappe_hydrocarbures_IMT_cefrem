from enum import Enum


class EnumLoss(Enum,str):
    MulticlassnonExlusivCrossentropy = "multiclassnonexlusivcrossentropy"
    BinaryCrossentropy = "binarycrossentropy"
    MSE = "mse"