"""Contains enumeration EnumBalance"""

from enum import Enum


class EnumBalance(str, Enum):
    BalanceClasses2 = "balanceclasses2"
    """Exclude patches with classes other than the other category"""
    BalanceClasses1 = "balanceclasses1"
    """Exclude patches with only the other category"""
    NoBalance = "nobalance"
    """Does not filter pataches"""
