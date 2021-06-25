"""
Classes that manage the dataset class balancing.

Currently one balancing type is supported:
- BalanceClasses1: balance classes by excluding patches where there is only the other class

The filter method of this class is called on the original classification patch label (see ClassificationPatch class) once they are generated.

We do not know in advance the repartition of each class at this time

The NoBalance class is added to give access to a similar interface even if we do not want to make any balancing step
"""