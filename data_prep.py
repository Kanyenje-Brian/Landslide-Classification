import numpy as np
import pandas as pd
import io
from scipy.stats import kurtosis, skew


def load_image(file_content):
    with io.BytesIO(file_content) as f:
        img = np.load(f)
    return img.astype('float32')

def extract_additional_features(img_array):
    """Calculates derived indices like NDVI and NDWI."""
    red, green, nir = img_array[..., 0], img_array[..., 1], img_array[..., 3]
    epsilon = 1e-7
    ndvi = (nir - red) / (nir + red + epsilon)
    ndwi = (green - nir) / (green + nir + epsilon)
    return np.concatenate([img_array, ndvi[..., np.newaxis], ndwi[..., np.newaxis]], axis=-1)



def create_statistical_features_df(image_features_array):
   
    n_samples = image_features_array.shape[0]
    stats_df = pd.DataFrame(index=np.arange(n_samples))

    band_names = [f'band{i+1}' for i in range(12)] + ['ndvi', 'ndwi']

    # Calculate the mean brightness across RGB channels for each image
    print("Adding cloud score (mean brightness) to statistical features...")
    rgb_data = image_features_array[:, :, :, :3]
    stats_df['cloud_score_brightness'] = np.mean(rgb_data, axis=(1, 2, 3))

    print(f"\nGenerating statistical features for {n_samples} samples...")
    for i, name in enumerate(band_names):
        channel_data = image_features_array[:, :, :, i]
        stats_df[f'{name}_mean'] = channel_data.mean(axis=(1, 2))
        stats_df[f'{name}_std']  = channel_data.std(axis=(1, 2))
        stats_df[f'{name}_min']  = channel_data.min(axis=(1, 2))
        stats_df[f'{name}_max']  = channel_data.max(axis=(1, 2))
        stats_df[f'{name}_kurt'] = kurtosis(channel_data, axis=(1, 2), fisher=True)
        stats_df[f'{name}_skew'] = skew(channel_data, axis=(1, 2))
    
    print("...statistical features created successfully.")
    return stats_df
