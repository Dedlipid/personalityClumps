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
    - A 2D numpy array where each row contains the distances of a data point to each cluster center.
    """
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    distances = np.array([np.linalg.norm(data - center, axis=1) for center in cluster_centers]).T
    return distances

def clusterencodingdraw(encoding, names, k=4):
    """
    Embed each point in k-1 dimensional space by using its first k-1 distances as the values and label it by its name.
    Use Plotly to output an interactable 3D map for k = 4.
    
    Parameters:
    - encoding: A 2D numpy array where each row contains distances of a data point to each cluster center.
    - names: A list of names corresponding to each data point.
    - k: The number of clusters used in encoding.
    """
    if k != 4:
        raise ValueError("This function currently supports only k=4 for 3D visualization.")
    
    fig = px.scatter_3d(
        x=encoding[:, 0], 
        y=encoding[:, 1], 
        z=encoding[:, 2], 
        text=names,
        title="3D Cluster Encoding Visualization"
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
    encoding = kclusterencoding(data, k)
    clusterencodingdraw(encoding, names, k)

