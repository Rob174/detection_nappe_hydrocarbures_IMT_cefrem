from main.FolderInfos import FolderInfos
from main.src.params_saver.markdown_saver import Markdown_saver
import inspect
import json


class JSON_saver:
    def __init__(self,markdown_saver: Markdown_saver):
        members = inspect.getmembers(markdown_saver) # Get attributes and methods of the class
        attributes = list(filter(lambda x: inspect.ismethod(x[1]) is False and x[0][:5] == "attr_", members))
        self.data = {}
        for attr,value in attributes:
            self.data[attr] = value
        self.data = str(json.dumps(self.data,indent=4))

    def __call__(self, tensorboard_summary_writer):
        tensorboard_summary_writer.add_text("JSON_summary", self.data, global_step=0)

if __name__ ==  "__main__":
    FolderInfos.init(test_without_data=True)
    markdown_saver = Markdown_saver(number_imgs_used=1000, class_mappings={"other": 0, "spill": 1, "seep": 2},
                       grid_size=500, preprocessing="NR_Cal_ML_EC_dB", loss="CrossEntropyLoss",
                       model_name="efficientnetv4",
                       metrics=["Accuracy"], optimizer={"name": "SGD", "lr": 0.001, "momentum": 0.9}
                       )
    print(JSON_saver(markdown_saver).data)