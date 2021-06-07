import subprocess
import inspect

from main.FolderInfos import FolderInfos
from torch.utils.tensorboard import SummaryWriter


class Markdown_saver:
    def __init__(self, number_imgs_used: int, class_mappings: dict,
                 grid_size: int,
                 loss: str, metrics: list, optimizer: dict, model_name: str,
                 preprocessing: str = None, data_augmentations: list = None,
                 ):
        with open(FolderInfos.input_data_folder + "excluded_data_from_cache.txt", "r") as fp:
            self.attr_images_excluded = fp.readlines()
        self.attr_number_imgs_used: int = number_imgs_used
        self.attr_class_mappings: dict = class_mappings
        self.attr_grid_size_px: int = grid_size
        self.mappings_preprocessing = {"NR": "ThermaloiseRemoval",
                                       "Cal": "Calibration",
                                       "ML": "Multilook",
                                       "EC": "Ellipsoid-CorrectionGG",
                                       "dB": "LinearToFromdB"}
        self.attr_preprocessing: str = preprocessing  # in the NR_Cal_ML_EC_dB format
        self.attr_data_augmentations: list = data_augmentations
        self.attr_loss: str = loss
        self.attr_metrics: list = metrics
        self.attr_optimizer: dict = optimizer
        self.attr_model_name: str = model_name
        self.attr_hash_comit: str = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip()
        self.attr_id = FolderInfos.id

        text = f"""# Logs training {FolderInfos.id} commit {self.attr_hash_comit}
        
## Data:

{self.attr_number_imgs_used} images used with a {self.attr_grid_size_px} by {self.attr_grid_size_px} px gridsize\n\n"""
        if isinstance(self.attr_preprocessing,str) and self.attr_preprocessing != "":
            self.attr_preprocessing = list(map(lambda x:self.mappings_preprocessing[x],self.attr_preprocessing.split("_")))
            text += "Preprocessing applied: "
            for preprocess in self.attr_preprocessing:
                text += f" {preprocess} ;"
        else:
            text += f"No preprocessing applied"
        text += "\n\n"

        if isinstance(self.attr_data_augmentations,list):
            text += "Data augmentations used:\n\n"
            for augm in self.attr_data_augmentations:
                text += f"- {augm}\n\n"
        else:
            text += "No augmentations used\n\n"

        text += "Images excluded from the dataset cache:\n\n"
        for img in self.attr_images_excluded:
            text += f"- {img}\n"
        text += "\n\n"

        text += f"""\n## Model\n\nModel {self.attr_model_name} trained with the {self.attr_optimizer["name"]} optimizer with """
        text += ", ".join([f"{param_name}: {param_value}" for param_name, param_value in self.attr_optimizer.items() if
                           param_name != "name"]) + "\n\n"
        text += f"Metrics used: " + ",".join(metrics) + "\n\n"

        self.data: str = text

    def __call__(self, tensorboard_summary_writer: SummaryWriter):
        tensorboard_summary_writer.add_text("Markdown_summary", self.data, global_step=0)

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    print(
        Markdown_saver(number_imgs_used=1000,class_mappings={"other": 0,"spill": 1,"seep": 2},
                         grid_size=500,preprocessing="NR_Cal_ML_EC_dB",loss="CrossEntropyLoss",model_name="efficientnetv4",
                         metrics=["Accuracy"],optimizer={"name":"SGD","lr":0.001, "momentum":0.9}
                         )(SummaryWriter(log_dir=FolderInfos.data_test))
                                )