import numpy as np

from main.src.param_savers.BaseClass import BaseClass


class BatchMaker(BaseClass):
    def __init__(self, batch_size: int, num_elems_gen: int = 4):
        self.attr_batch_size = batch_size
        self.num_elements_generated = num_elems_gen

    def batch(self, generator_vals):
        to_stack = [[] for _ in range(self.num_elements_generated)]
        for sample in generator_vals:
            for i, elem in enumerate(sample):
                to_stack[i].append(elem)

            if len(to_stack) == self.attr_batch_size:
                stacked_return = []
                for list in to_stack:
                    if isinstance(list[0], np.ndarray):
                        stacked_return.append(np.stack(list, axis=0))
                    elif isinstance(list[0], str):
                        stacked_return.append(list)
                    else:
                        raise TypeError(f"Unsupported type of data yielded {type(list[0])}")
                return stacked_return
