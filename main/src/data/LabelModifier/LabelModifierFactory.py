from typing import Tuple

from main.src.data.LabelModifier.LabelModifier0 import LabelModifier0
from main.src.data.LabelModifier.LabelModifier1 import LabelModifier1
from main.src.data.LabelModifier.LabelModifier2 import LabelModifier2
from main.src.data.LabelModifier.NoLabelModifier import NoLabelModifier
from main.src.enums import EnumLabelModifier, EnumClasses


class LabelModifierFactory:
    def create(self, label_modifier: EnumLabelModifier, class_mapping,
               classes_to_use: Tuple[EnumClasses] = (EnumClasses.Seep, EnumClasses.Spill)):
        if label_modifier == EnumLabelModifier.NoLabelModifier:
            return NoLabelModifier(original_class_mapping=class_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier0:
            return LabelModifier0(class_mapping=class_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier1:
            return LabelModifier1(classes_to_use=classes_to_use,
                                                      original_class_mapping=class_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier2:
            return LabelModifier2(classes_to_use=classes_to_use,
                                                      original_class_mapping=class_mapping)
        else:
            raise NotImplementedError(f"{label_modifier} is not implemented")
