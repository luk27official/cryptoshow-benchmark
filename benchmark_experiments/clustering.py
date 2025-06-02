import numpy as np
from sklearn.cluster import DBSCAN


def compute_clusters(
    points: list[list[float]], prediction_scores: list[float], eps: float = 5.0, min_samples: int = 3
) -> np.ndarray:
    """
    Compute clusters based on the given points and prediction scores.

    Args:
        points (list[list[float]]): A list of points, where each point is a list of 3 coordinates [x, y, z].
        prediction_scores (list[float]): A list of prediction scores corresponding to each point.
        eps (float): The maximum distance between two samples for one to be considered in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        np.ndarray: An array of cluster labels for each point. Points with no cluster are labeled as -1.
    """

    points_array = np.array(points)
    scores_array = np.array(prediction_scores).reshape(-1, 1)
    stacked = np.hstack((points_array, scores_array))  # Combine coordinates with scores

    high_score_mask = stacked[:, 3] > 0.65
    high_score_points = stacked[high_score_mask][:, :3]  # Extract only (x, y, z) coordinates

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(high_score_points)

    # Initialize all labels to -1
    all_labels = -1 * np.ones(len(points), dtype=int)
    # Assign cluster labels to high score points
    all_labels[high_score_mask] = labels
    labels = all_labels

    # After labelling the clusters, for each cluster, compute the centroid
    # For all points in that cluster, get the furthest point from the centroid
    # and assign all points in a radius of 0.5x the distance of the furthest point
    # the same label as the cluster
    for cluster_label in np.unique(labels):
        if cluster_label == -1:
            continue
        cluster_points = points_array[labels == cluster_label]
        centroid = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        furthest_point = np.max(distances)
        radius = 0.75 * furthest_point
        within_radius_mask = np.linalg.norm(points_array - centroid, axis=1) <= radius
        labels[within_radius_mask & labels == -1] = cluster_label

    return labels
