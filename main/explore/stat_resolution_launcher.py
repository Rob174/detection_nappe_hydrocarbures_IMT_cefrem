from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory

if __name__ == "__main__":
    FolderInfos.init(test_without_data=False)
    dataset_factory = DatasetFactory(dataset_name="sentinel1", usage_type="classification", patch_creator="fixed_px",
                                     patch_padding="no", grid_size=500, input_size=256)
    length = len(dataset_factory)
    print(f"{length} items in this dataset")
    for id in range(length):
        input, output = dataset_factory[id]
        if id % int(length * 0.1) == 0:
            print(f"{id / length * 100:.2f} done")
    dataset_factory.save_stats()
    end = 0