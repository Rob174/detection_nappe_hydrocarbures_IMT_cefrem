from main.src.param_savers.BaseClass import BaseClass


class BalanceClasses1(BaseClass):
    def __init__(self,classes_indexes, margin=10):
        """

        @param classes_indexes:
        @param margin: specify the max difference of number of samples provided with two classes of the possible classes
        """
        self.attr_margin = margin
        self.attr_number_of_classes = {k:0 for k in classes_indexes}
        self.attr_number_accepted = {k:0 for k in classes_indexes}
        self.attr_prct_accepted = {k:0 for k in classes_indexes}
        self.attr_name = self.__class__.__name__
        self.attr_global_name = "balance"

    def filter(self,classification_label):
        """

        @param classification_label: label with ones of a class is on the image and 0 if not. !! Must be provided by ClassificationPatch make_classification_label method for the shape of the labels (with full details)
        @return: bool, tell if the sample is accepted or rejected
        """
        tmp_if_accepted_dico = {k:v for k,v in self.attr_prct_accepted}
        for k in self.attr_number_of_classes:
            self.attr_number_of_classes[k] += int(classification_label[k])
            tmp_if_accepted_dico[k] += int(classification_label[k])
        reject = True
        num_classes = tmp_if_accepted_dico.values()
        if max(num_classes)-min(num_classes) >= self.attr_margin:
            reject = False
            for k in self.attr_number_of_classes:
                self.attr_number_accepted[k] += int(classification_label[k])
                self.attr_prct_accepted[k] = self.attr_number_accepted[k] / self.attr_number_of_classes[k]
        return reject