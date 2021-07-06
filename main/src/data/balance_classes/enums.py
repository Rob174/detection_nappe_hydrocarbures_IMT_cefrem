from enum import Enum


class EnumBalance(Enum,str):
    BalanceClasses1 = "balanceclasses1"
    """Exclude patches with only the other category"""
    NoBalance = "nobalance"
    """Does not filter pataches"""