"""Script to launch with(out) arguments to train the attr_model and save the results"""

import subprocess
import sys


sys.path.append(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem")
sys.path.append(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\main")
sys.path.append(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\main\src")
sys.path.append(r"C:\Users\robin\Documents\projets")

from main.src.analysis.analysis.RGB_Overlay2 import RGB_Overlay2
from main.src.training.early_stopping.EarlyStopping import EarlyStopping
from main.src.training.periodic_model_saver.ModelSaver1 import ModelSaver1
from main.src.data.Standardizer.NoStandardizer import NoStandardizer
from main.src.parsers.ParserGenerateFilteredCache import ParserGenerateFilteredCache

# from main.src.training.Trainers.TrainerGenerateCache import TrainerGenerateCache
from main.src.training.Trainers.TrainerGenerateCache import TrainerGenerateCache
from main.src.training.metrics.metrics_factory import MetricsFactory

from main.src.training.metrics.loss_factory import LossFactory
from main.src.training.optimizers_factory import OptimizersFactory
from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.parsers.ParserClassificationCache import ParserClassificationCache
from main.src.models.ModelFactory import ModelFactory
from main.src.param_savers.saver0 import Saver0
from main.src.enums import EnumLabelModifier
from main.src.enums import EnumGitCheck

if __name__ == "__main__":
    FolderInfos.init()
    saver = Saver0(FolderInfos.base_filename + "parameters.json")
    parser = ParserGenerateFilteredCache()
    saver(parser)
    arguments = parser()
    saver["commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip()
    changes = subprocess.check_output(['git', 'diff', '--name-only', "--", "*.py"]).decode("ascii").split("\n")[:-1]
    if arguments.no_security == EnumGitCheck.GITCHECK and len(changes):
        changes_str = "\n".join(list(map(lambda x: '\t- ' + x, changes)))
        input(
            f"There are {len(changes)} uncommitted python files:\n{changes_str}\n Are you sure you want to continue ?")
    print("✔ Commit ok")

    dataset = DatasetFactory(dataset_name=arguments.attr_dataset,
                             usage_type=arguments.usage_type,
                             grid_size=arguments.grid_size,
                             input_size=arguments.input_size,
                             exclusion_policy=arguments.patch_exclude_policy,
                             exclusion_policy_threshold=arguments.patch_exclude_policy_threshold,
                             classes_to_use=arguments.classes,
                             balance=arguments.balance,
                             augmenter_img=arguments.augmenter_img,
                             augmentations_img=arguments.augmentations_img,
                             augmentation_factor=arguments.augmentation_factor,
                             other_class_adder=arguments.other_class_adder,
                             interval=arguments.interval,
                             choose_dataset="patch"
                             )
    saver["date"] = FolderInfos.id

    num_classes = len(arguments.classes) if arguments.attr_dataset != EnumLabelModifier.LabelModifier2 else 1
    model = ModelFactory(model_name=arguments.attr_model, num_classes=num_classes, freeze=arguments.freeze)
    optimizer = OptimizersFactory(model, name=arguments.optimizer,
                                  lr=arguments.lr, eps=arguments.eps)
    loss = LossFactory(usage_type=arguments.usage_type, preference=arguments.loss_preference,optimizer=optimizer)
    metrics = MetricsFactory("accuracy_classification-0.25", "accuracy_classification-0.1", "mae", "accuracy_threshold-0.5")
    model_saver = ModelSaver1(loss, loss.attr_loss)
    early_stopping = EarlyStopping(loss, name_metric_chosen=loss.attr_loss, patience=5)
    saver.save()
    try:
        standardizer = dataset.attr_dataset.attr_standardizer
    except AttributeError:
        standardizer = NoStandardizer()
    rgb_overlay = RGB_Overlay2(
        standardizer=standardizer
    )
    print("start")
    TrainerGenerateCache(
        batch_size=arguments.batch_size,
        num_epochs=arguments.num_epochs,
        tr_prct=0.7,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        early_stopping=early_stopping,
        model_saver=model_saver,
        saver=saver,
        eval_step=arguments.eval_step,
        debug=arguments.debug,
        rgb_overlay=rgb_overlay
    )(name="filtered_cache")
    print("end")
