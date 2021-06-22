from main.src.param_savers.BaseClass import BaseClass


class NoBalance(BaseClass):
    def __init__(self):
        """
        Class used to indicate that no balancing operation has been applied
        """
        super(NoBalance, self).__init__()
        self.attr_name = self.__class__.__name__
        self.attr_global_name = "balance"
    def filter(self,classification_label):
        return False # we always accept a sample