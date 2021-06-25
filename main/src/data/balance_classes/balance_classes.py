from main.src.param_savers.BaseClass import BaseClass


class BalanceClasses1(BaseClass):
    def __init__(self,classes_indexes, margin=10):
        """Balance classes by keeping an image if the classes on it are not overrepresented

        Args:
            classes_indexes: classes indexes to consider
            margin: specify the max difference of number of samples provided with two classes of the possible classes
        """
        self.attr_margin = margin
        self.attr_number_of_classes = {k:0 for k in classes_indexes}
        self.attr_number_accepted = {k:0 for k in classes_indexes}
        self.attr_prct_accepted = {k:0 for k in classes_indexes}
        self.attr_name = self.__class__.__name__ # save the name of the class used for reproductibility purposes
        self.attr_global_name = "balance" # save a more compehensible name

    def filter(self,classification_label):
        """method called during training to know if we have to filter this sample or not based on its classification_label

        Args:
            classification_label:  label with ones of a class is on the image and 0 if not. !! Must be provided by ClassificationPatch make_classification_label method for the shape of the labels (with full details)

        Returns:
            bool, tell if the sample is accepted or rejected

        """
        # We test the effect of adding this new sample regarding the number of already seen classes
        tmp_if_accepted_dico = {k:v for k,v in self.attr_number_accepted.items()}
        for k in self.attr_number_of_classes:
            self.attr_number_of_classes[k] += int(classification_label[k])
            tmp_if_accepted_dico[k] += int(classification_label[k])
        reject = True
        num_classes = tmp_if_accepted_dico.values()
        # if adding this sample allows to keep a difference between the most seen and the least seen classes below the margin, we accept this new sample
        if max(num_classes)-min(num_classes) <= self.attr_margin:
            reject = False
            # and we update statistics
            for k in self.attr_number_of_classes:
                self.attr_number_accepted[k] += int(classification_label[k])
                self.attr_prct_accepted[k] = self.attr_number_accepted[k] / self.attr_number_of_classes[k] if self.attr_number_of_classes[k] != 0 else 0
        return reject