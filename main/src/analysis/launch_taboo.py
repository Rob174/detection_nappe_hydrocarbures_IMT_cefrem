"""Script to launch after running the first cell of the analysis.ipynb notebook to view all trainings available for analysis"""

import pandas as pd
import tabloo

from main.FolderInfos import FolderInfos

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    l_path = FolderInfos.root_folder.split(FolderInfos.separator) + ["main", "src", "analysis", "data_taboo.pkl"]
    path_cache = FolderInfos.separator.join(l_path)
    df = pd.read_pickle(path_cache)
    tabloo.show(df)
