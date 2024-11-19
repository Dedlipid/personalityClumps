import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px


def kclusterencoding(data, k=4):
    """
    Find k clusters, return an array of arrays that for each data point have its distance to each cluster.

    Parameters:
    - data: A 2D numpy array where each row is a data point.
    - k: The number of clusters to form.

    Returns:
    - A tuple containing:
      - A 2D numpy array where each row contains the distances of a data point to each cluster center.
      - A 1D numpy array with the index of the closest cluster for each data point.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    distances = np.array(
        [np.linalg.norm(data - center, axis=1) for center in cluster_centers]
    ).T
    closest_clusters = np.argmin(distances, axis=1)
    return distances, closest_clusters


def clusterencodingdraw(encoding, closest_clusters, names, k=4):
    """
    Embed each point in k-1 dimensional space by using its first k-1 distances as the values and label it by its name.
    Use Plotly to output an interactable 3D map for k = 3 or k = 4, coloring points by their closest cluster.

    Parameters:
    - encoding: A 2D numpy array where each row contains distances of a data point to each cluster center.
    - closest_clusters: A 1D numpy array with the index of the closest cluster for each data point.
    - names: A list of names corresponding to each data point.
    - k: The number of clusters used in encoding.
    """
    if k not in [3, 4]:
        raise ValueError(
            "This function currently supports only k=3 or k=4 for 3D visualization."
        )

    fig = px.scatter_3d(
        x=encoding[:, 0],
        y=encoding[:, 1],
        z=encoding[:, 2],
        text=names,
        color=closest_clusters.astype(str),  # Color by cluster index
        title="3D Cluster Encoding Visualization",
    )
    fig.write_html("cluster_encoding.html")
    fig.show()


def perform_encoding_and_draw(data, names, k=4):
    """
    Perform k-cluster encoding and draw the 3D visualization.

    Parameters:
    - data: A 2D numpy array where each row is a data point.
    - names: A list of names corresponding to each data point.
    - k: The number of clusters to form and visualize.
    """
    encoding, closest_clusters = kclusterencoding(data, k)
    clusterencodingdraw(encoding, closest_clusters, names, k)
