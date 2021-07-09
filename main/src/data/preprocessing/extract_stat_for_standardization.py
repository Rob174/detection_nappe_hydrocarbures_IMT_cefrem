import h5py
import numpy as np
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from main.FolderInfos import FolderInfos

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    with h5py.File(FolderInfos.input_data_folder + "images_preprocessed.hdf5", "r") as cache:
        dico_stats = {'sum_all_vals': -1055176831312.0, 'total_num_px': 94929199354}
        # Compute the mean of all the attr_dataset
        progress = Progress(
            TextColumn("{task.fields[name]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn()
        )
        # with progress:
        #     task = progress.add_task("progress", name="[bold blue]Iteration", total=len(cache))
        #     dico_stats = {"sum_all_vals": 0, "total_num_px": 0}
        #     for k,v in cache.items():
        #         v = np.array(v)
        #         dico_stats["sum_all_vals"] += np.sum(v)
        #         dico_stats["total_num_px"] += v.shape[0] * v.shape[1]
        #         progress.update(task, advance=1)
        print(dico_stats["sum_all_vals"] / dico_stats["total_num_px"])
        with progress:
            task = progress.add_task("progress", name="[bold blue]Iteration", total=len(cache))
            dico_stats["sum_squared_diff_mean"] = 0
            for k, v in cache.items():
                v = np.array(v)
                diff_mean = v - (dico_stats["sum_all_vals"] / dico_stats["total_num_px"])
                dico_stats["sum_squared_diff_mean"] += np.sum(np.power(diff_mean, 2))
                progress.update(task, advance=1)
        print(dico_stats)
    import json

    with open(FolderInfos.input_data_folder + "pixel_stats.json", "w") as fp:
        json.dump(dico_stats, fp, indent=4)
