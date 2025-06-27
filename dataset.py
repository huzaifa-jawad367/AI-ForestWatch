import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

def get_indices(arr):
    if arr.ndim != 3 or arr.shape[2] < 7:
        raise ValueError("Input array must be 3-dimensional with at least 7 channels.")
    
    bands = {
        "ndvi": (arr[:, :, 4] - arr[:, :, 3]) / (arr[:, :, 4] + arr[:, :, 3] + 1e-7),
        "evi": 2.5 * (arr[:, :, 4] - arr[:, :, 3]) / (arr[:, :, 4] + 6 * arr[:, :, 3] - 7.5 * arr[:, :, 1] + 1),
        "savi": 1.5 * (arr[:, :, 4] - arr[:, :, 3]) / (arr[:, :, 4] + arr[:, :, 3] + 0.5),
        "msavi": 0.5 * (2 * arr[:, :, 4] + 1 - np.sqrt((2 * arr[:, :, 4] + 1) ** 2 - 8 * (arr[:, :, 4] - arr[:, :, 3]))),
        "ndmi": (arr[:, :, 4] - arr[:, :, 5]) / (arr[:, :, 4] + arr[:, :, 5] + 1e-7),
        "nbr": (arr[:, :, 4] - arr[:, :, 6]) / (arr[:, :, 4] + arr[:, :, 6] + 1e-7),
        "nbr2": (arr[:, :, 5] - arr[:, :, 6]) / (arr[:, :, 5] + arr[:, :, 6] + 1e-7),
    }
    for name in bands:
        value = np.nan_to_num(bands[name])
        arr = np.dstack((arr, value))
    return arr

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val + 1e-7)

class SlidingWindowDataset(Dataset):
    def __init__(self, pickle_dir, window_size=128, stride=64, reduce_indices=False):
        self.pickle_dir = pickle_dir
        self.window_size = window_size
        self.stride = stride
        self.reduce_indices = reduce_indices
        self.processed_images, self.processed_masks = self._process_data()

    def _process_data(self):
        processed_images = []
        processed_masks = []
        
        for file_name in os.listdir(self.pickle_dir):
            if file_name.endswith('.pkl'):
                with open(os.path.join(self.pickle_dir, file_name), 'rb') as f:
                    img, mask = pickle.load(f, encoding='latin1')
                
                if img.ndim == 3 and img.shape[2] >= 7:
                    img = get_indices(img)
                    img = normalize_image(img)
                    h, w, _ = img.shape
                    for i in range(0, h - self.window_size + 1, self.stride):
                        for j in range(0, w - self.window_size + 1, self.stride):
                            window_img = img[i:i + self.window_size, j:j + self.window_size]
                            window_mask = mask[i:i + self.window_size, j:j + self.window_size]
                            class_0_ratio = np.sum(window_mask == 0) / window_mask.size
                            class_1_ratio = np.sum(window_mask == 1) / window_mask.size
                            class_2_ratio = np.sum(window_mask == 2) / window_mask.size
                            if class_0_ratio < 0.5:
                                if class_2_ratio > 0.4:
                                    # Augment the image by rotating it 90 degrees 3 times
                                    for _ in range(3):
                                        window_img = np.rot90(window_img).copy()
                                        window_mask = np.rot90(window_mask).copy()
                                        processed_images.append(window_img)
                                        processed_masks.append(window_mask)
                                else:
                                    processed_images.append(window_img)
                                    processed_masks.append(window_mask)
                else:
                    print(f"Skipping image with shape {img.shape} in file {file_name}")
        
        return processed_images, processed_masks

    def __len__(self):
        return len(self.processed_images)

    def __getitem__(self, idx):
        image = self.processed_images[idx]
        mask = self.processed_masks[idx]
        
        if mask.dtype == np.uint16:
            mask = mask.astype(np.int64)
            
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to CxHxW
        mask = torch.tensor(mask, dtype=torch.long)
        
        if self.reduce_indices:
            mask = mask - 1
            mask[mask == -1] = 255
            
        encoded_data = {
            'pixel_values': image,
            'labels': mask
        }

        return encoded_data