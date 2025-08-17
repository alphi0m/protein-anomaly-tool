# logic/clustering_utils.py

import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, SpectralClustering
import plotly.graph_objs as go


def dbscan_clustering(pca_df, eps=0.25, min_samples=15):
    """
    Applica DBSCAN su pca_df.
    pca_df: DataFrame con colonne PC1, PC2, PC3, ...
    Restituisce il DataFrame con colonna 'Cluster'.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(pca_df)
    pca_df['Cluster'] = clusters
    return pca_df


def optics_clustering(pca_df, min_samples=5, xi=0.01, min_cluster_size=0.1):
    """
    Applica OPTICS su pca_df.
    """
    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    clusters = optics.fit_predict(pca_df)
    pca_df['Cluster'] = clusters
    return pca_df


def spectral_clustering(pca_df, n_clusters=3, affinity='nearest_neighbors', n_neighbors=10):
    """
    Applica Spectral Clustering su pca_df.
    """
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors,
        assign_labels='kmeans'
    )
    clusters = spectral.fit_predict(pca_df)
    pca_df['Cluster'] = clusters
    return pca_df


def plot_clusters_3d_plotly(pca_df, title="Clustering 3D"):
    """
    Restituisce un grafico Plotly 3D dei cluster su PC1, PC2, PC3.
    """
    pc_cols = [c for c in pca_df.columns if c.startswith("PC")]
    fig = go.Figure()

    for cluster in pca_df['Cluster'].unique():
        cluster_points = pca_df[pca_df['Cluster'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_points[pc_cols[0]],
            y=cluster_points[pc_cols[1]] if len(pc_cols) > 1 else cluster_points[pc_cols[0]],
            z=cluster_points[pc_cols[2]] if len(pc_cols) > 2 else cluster_points[pc_cols[0]],
            mode='markers',
            marker=dict(size=4),
            name=f'Cluster {cluster}'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=pc_cols[0],
            yaxis_title=pc_cols[1] if len(pc_cols) > 1 else '',
            zaxis_title=pc_cols[2] if len(pc_cols) > 2 else ''
        ),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font_color="white"
    )
    return fig