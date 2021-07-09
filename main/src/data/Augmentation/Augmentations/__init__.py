"""
Contains one class per augmentation with the following mandatory ⚠️ structure:
>>> class MyAugmentation:
...     @staticmethod
...     def compute_random_augment(image: np.ndarray, annotation: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
...        # code ...
...        return image,annotation
"""
