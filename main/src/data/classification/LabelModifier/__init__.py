"""Change the original classification with 3 labels (other, seep, spill) according to 3 possible algorithms:

- NoLabelModifier: Change nothing, only for common interface purpose
- LabelModifier1: Allow to use less classes than other, seep, spill and to change their order.
- LabelModifier2: Merge multiple labels the classes to use and indicate the presence or absence of one of these classes
"""