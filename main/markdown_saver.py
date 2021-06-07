import subprocess
import os

from main.FolderInfos import FolderInfos
# from torch.utils.tensorboard import SummaryWriter


class Markdown_saver:
    def __init__(self, number_imgs_used: int, class_mappings: dict,
                 grid_size: int,
                 loss: str, metrics: list, optimizer: dict, model_name: str,
                 preprocessing: str = None, data_augmentations: list = None,
                 ):
        with open(FolderInfos.input_data_folder + "excluded_data_from_cache.txt", "r") as fp:
            self.images_excluded = fp.readlines()
        self.number_imgs_used: int = number_imgs_used
        self.class_mappings: dict = class_mappings
        self.grid_size: int = grid_size
        self.preprocessing: str = preprocessing  # in the NR_Cal_ML_EC_dB format
        self.data_augmentations: list = data_augmentations
        self.loss: str = loss
        self.metrics: list = metrics
        self.optimizer: dict = optimizer
        self.model_name: str = model_name
        self.hash_comit: str = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip()

        text = f"""# Logs training {FolderInfos.id} commit {self.hash_comit}
        
## Data:

{self.number_imgs_used} images used with a {self.grid_size} by {self.grid_size} px gridsize\n\n"""
        self.mappings_preprocessing = {"NR": "ThermaloiseRemoval",
                                       "Cal": "Calibration",
                                       "ML": "Multilook",
                                       "EC": "Ellipsoid-CorrectionGG",
                                       "dB": "LinearToFromdB"}
        if self.preprocessing != {}:
            text += "Preprocessing applied: "
            for preprocess in self.preprocessing.split("_"):
                text += f" {self.mappings_preprocessing[preprocess]} ;"
        else:
            text += f"No preprocessing applied"
        text += "\n\n"

        if isinstance(self.data_augmentations,list):
            text += "Data augmentations used:\n\n"
            for augm in self.data_augmentations:
                text += f"- {augm}\n\n"
        else:
            text += "No augmentations used\n\n"

        text += "Images excluded from the dataset cache:\n\n"
        for img in self.images_excluded:
            text += f"- {img}\n"
        text += "\n\n"

        text += f"""\n## Model\n\nModel {self.model_name} trained with the {self.optimizer["name"]} optimizer with """
        text += ", ".join([f"{param_name}: {param_value}" for param_name, param_value in self.optimizer.items() if
                           param_name != "name"]) + "\n\n"
        text += f"Metrics used: " + ",".join(metrics) + "\n\n"

        self.markdown: str = text

    def __call__(self, tensorboard_summary_writer):
        tensorboard_summary_writer.add_text("Markdown_summary", self.markdown, global_step=0)

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    print(Markdown_saver(1000,class_mappings={"other": 0,"spill": 1,"seep": 2},
                         grid_size=500,preprocessing="NR_Cal_ML_EC_dB",loss="CrossEntropyLoss",model_name="efficientnetv4",
                         metrics=["Accuracy"],optimizer={"name":"SGD","lr":0.001, "momentum":0.9}
                         ).markdown)