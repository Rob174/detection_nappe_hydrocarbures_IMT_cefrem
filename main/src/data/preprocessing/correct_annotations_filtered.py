import json
import numpy as np

from h5py import File
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from main.FolderInfos import FolderInfos
from main.src.data.Annotations.PointAnnotations import PointAnnotations

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    with open(FolderInfos.input_data_folder+"filtered_img_infos.json","r") as fp:
        dico_infos = json.load(fp)
    cache_annotations_pts = PointAnnotations()
    with File(FolderInfos.input_data_folder+"filtered_cache_annotations.hdf5","w") as cache_annotations:
        with Progress(
            TextColumn("{task.fields[name]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TextColumn("[bold blue]status: {task.fields[status]}", justify="right"),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn()) as progress:
            task = progress.add_task("processed", name="[red]Imgs processed", total=len(dico_infos),status=0)
            for i,(name,dico) in enumerate(dico_infos.items()):
                transformation_matrix = np.array(dico["transformation_matrix"],dtype=np.float32)
                source_img = dico["source_img"]
                segmentation_map = cache_annotations_pts[source_img,transformation_matrix,256]
                cache_annotations.create_dataset(name, shape=segmentation_map.shape, dtype='i',
                                                   data=segmentation_map)
                progress.update(task, advance=1, status=i)