import tabloo
import pandas as pd

if __name__ == "__main__":
    path_cache = "C:/Users/robin/Documents/projets/detection_nappe_hydrocarbures_IMT_cefrem/main/src/analysis/data_taboo.pkl"
    df = pd.read_pickle(path_cache)
    tabloo.show(df)