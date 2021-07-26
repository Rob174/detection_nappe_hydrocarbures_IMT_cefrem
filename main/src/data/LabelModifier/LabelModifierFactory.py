from typing import Tuple

from main.src.data.LabelModifier.LabelModifier0 import LabelModifier0
from main.src.data.LabelModifier.LabelModifier1 import LabelModifier1
from main.src.data.LabelModifier.LabelModifier2 import LabelModifier2
from main.src.enums import EnumLabelModifier, EnumClasses


class LabelModifierFactory:
    def create(self,label_modifier: EnumLabelModifier, class_mapping,
               classes_to_use: Tuple[EnumClasses] = (EnumClasses.Seep, EnumClasses.Spill)):
        if label_modifier == EnumLabelModifier.NoLabelModifier:
        elif label_modifier == EnumLabelModifier.LabelModifier1:
            self.attr_label_modifier = LabelModifier0(class_mapping=self.attr_label_dataset.attr_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier1:
            self.attr_label_modifier = LabelModifier1(classes_to_use=classes_to_use,
                                                      original_class_mapping=self.attr_label_dataset.attr_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier2:
            self.attr_label_modifier = LabelModifier2(classes_to_use=classes_to_use,
                                                      original_class_mapping=self.attr_label_dataset.attr_mapping)
        else:
            raise NotImplementedError(f"{label_modifier} is not implemented")
