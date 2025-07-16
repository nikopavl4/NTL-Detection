from typing import Optional

import numpy as np
from sklearn.cluster import KMeans


def gaussian_noise(data: np.ndarray, noise_std: Optional[float] = 0.01) -> np.ndarray:
    """
    Adds gaussian noise to the input data.

    Args:
        data (np.ndarray): Input features.
        noise_std (float, optional): Standard deviation of the Gaussian noise to be added. Defaults to 0.01

    Returns:
        np.ndarray: Augmented data with Gaussian noise.
    """
    noise = np.random.normal(0, noise_std, data.shape)
    augmented = data + noise
    augmented[augmented < 0] = 0.
    augmented[augmented > 1] = 1.
    return augmented


def cluster_features(data: np.ndarray, n_clusters: Optional[int] = 10) -> np.ndarray:
    """
    Adds clustering-based features to the input data.

    Args:
        data (np.ndarray): Input features.
        n_clusters (int, optional): Number of clusters to consider.

    Returns:
        np.ndarray: Augmented data with cluster-based features.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
    cluster_labels = kmeans.fit_predict(data)
    x_clustered = np.column_stack((data, cluster_labels))
    return x_clustered


def test_gaussian_noise():
    x = np.array([[0.2, 0.3, 1.], [0.15, 0.05, 0.002]])
    print(gaussian_noise(x))
    print(gaussian_noise(x))


def test_cluster_features():
    x = np.array([[0.2, 0.3, 1.], [0.15, 0.05, 0.002]])
    print(cluster_features(x, 2))


if __name__ == "__main__":
    test_gaussian_noise()
    test_cluster_features()
