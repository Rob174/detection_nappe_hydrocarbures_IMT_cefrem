"""Script to launch with(out) arguments to train the model and save the results"""

import json
import subprocess
import os, sys

from main.src.enums import EnumGitCheck

sys.path.append(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem")
sys.path.append(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\main")
sys.path.append(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\main\src")
sys.path.append(r"C:\Users\robin\Documents\projets")

from main.src.training.Trainer0 import Trainer0
from main.src.training.metrics_factory import MetricsFactory


from main.src.training.loss_factory import LossFactory
from main.src.training.optimizers_factory import OptimizersFactory
from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.parsers.Parser0 import Parser0
from main.src.models.ModelFactory import ModelFactory
from main.src.param_savers.saver0 import Saver0

if __name__ == "__main__":
    FolderInfos.init()
    saver = Saver0(FolderInfos.base_filename+"parameters.json")
    parser = Parser0()
    saver(parser)
    arguments = Parser0()()
    saver["commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip()
    changes = subprocess.check_output(['git', 'diff', '--name-only',"--","*.py"]).decode("ascii").split("\n")[:-1]
    if arguments.no_security == EnumGitCheck.GITCHECK and len(changes):
        changes_str = "\n".join(list(map(lambda x:'\t- '+x,changes)))
        input(f"There are {len(changes)} uncommitted python files:\n{changes_str}\n Are you sure you want to continue ?")
    dataset = DatasetFactory(dataset_name=arguments.dataset,
                             usage_type=arguments.usage_type,
                             patch_creator=arguments.patch,
                             grid_size=arguments.grid_size,
                             input_size=arguments.input_size,
                             exclusion_policy=arguments.patch_exclude_policy,
                             exclusion_policy_threshold = arguments.patch_exclude_policy_threshold,
                             classes_to_use=arguments.classes,
                             balance=arguments.balance,
                             augmenter_img=arguments.augmenter_img,
                             augmentations_img=arguments.augmentations_img,
                             augmenter_patch=arguments.augmenter_patch,
                             augmentations_patch=arguments.augmentations_patch,
                             augmentation_factor=arguments.augmentation_factor
                             )
    saver["date"] = FolderInfos.id
    saver(dataset)

    num_classes = len(arguments.classes.split(",")) if dataset.attr_dataset.__class__.__name__ != "ClassificationPatch2" else 1
    model = ModelFactory(model_name=arguments.model, num_classes=num_classes)
    saver(model)
    model = model()
    criterion = LossFactory(usage_type=arguments.usage_type, preference=arguments.loss_preference)
    optimizer = OptimizersFactory(model, name=arguments.optimizer,
                                  lr=arguments.lr, eps=arguments.eps)
    metrics = MetricsFactory("accuracy_classification-0.25","accuracy_classification-0.1","mae")
    saver(criterion)
    saver(optimizer)
    saver(metrics)
    optimizer_pytorch = optimizer()
    criterion_pytorch = criterion()
    saver.save()


    print("start")

    Trainer0(batch_size=arguments.batch_size,num_epochs=arguments.num_epochs,tr_prct=0.7,
             dataset=dataset,model=model,
             optimizer=optimizer_pytorch,loss=criterion_pytorch,metrics=metrics,saver=saver,
             eval_step=arguments.eval_step,debug=arguments.debug)()
    print("end")