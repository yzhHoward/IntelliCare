import os
import pickle
import random
import torch
import numpy as np
from umap import UMAP
from typing import List, Optional
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def cluster_embeddings(
    embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "dice", train_index=None
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((embeddings.shape[0] - 1) ** 0.5)
    model = UMAP(
        n_neighbors=100, n_components=dim, metric=metric, disconnection_distance=1, n_jobs=32
    )
    model.fit(embeddings[train_index] if train_index is not None else embeddings)
    reduced_embeddings = model.transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters
 

def GMM_cluster(embeddings: np.ndarray, n_components: int, train_index=None):
    gm = BayesianGaussianMixture(n_components=n_components, random_state=RANDOM_SEED, max_iter=300, n_init=5, init_params="k-means++")
    gm.fit(embeddings[train_index] if train_index is not None else embeddings)
    probs = gm.predict_proba(embeddings)
    return probs


def perform_clustering(dataset, 
    embeddings: np.ndarray, train_index=None, dim: int = 8, threshold: float = 0.05, n_components: int = 1000
) -> List[np.ndarray]:
    if not os.path.exists(f"export/{dataset}/reduced_embeddings.pkl"):
        print("Reducing embeddings...")
        reduced_embeddings = cluster_embeddings(embeddings, dim, train_index=train_index)
        assert np.isnan(reduced_embeddings).sum() == 0
        pickle.dump(reduced_embeddings, open(f"export/{dataset}/reduced_embeddings.pkl", "wb"))
        print("Reducing embeddings finished.")
    else:
        print(f"Loading reduced embeddings from reduced_embeddings")
        reduced_embeddings = pickle.load(open(f"export/{dataset}/reduced_embeddings.pkl", "rb"))
    if not os.path.exists(f"export/{dataset}/cluster_probs{n_components}.pkl"):
        print("Clustering...")
        cluster_probs = GMM_cluster(reduced_embeddings, n_components, train_index=train_index)
        pickle.dump(cluster_probs, open(f"export/{dataset}/cluster_probs{n_components}.pkl", "wb"))
        print("Clustering finished.")
        # exit()
    else:
        print(f"Loading clusters from cluster_probs{n_components}")
        cluster_probs = pickle.load(open(f"export/{dataset}/cluster_probs{n_components}.pkl", "rb"))
    labels = np.where((cluster_probs > threshold).sum(axis=0) > 0)[0]
    cluster_probs = torch.tensor(cluster_probs[:, labels])
    cluster_label = cluster_probs > threshold
    cluster_cnt = cluster_label.sum(dim=1)
    cluster_label_prob, cluster_label_list = torch.sort(cluster_probs, descending=True, dim=1)
    cluster_label_prob = [cluster_label_prob[i, :(cluster_label_prob[i] > threshold).sum()].tolist() for i in range(cluster_label_prob.shape[0])]
    cluster_label_list = [cluster_label_list[i, :len(cluster_label_prob[i])].tolist() for i in range(cluster_label_list.shape[0])]
    print(f"There are {cluster_cnt.shape[0]} samples, {cluster_probs.shape[1]} clusters in total, "
          f"{(cluster_cnt == 1).sum()} samples are assigned to one cluster, "
          f"{(cluster_cnt == 2).sum()} samples are assigned to two clusters, "
          f"{(cluster_cnt == 3).sum()} samples are assigned to three clusters, "
          f"{(cluster_cnt > 3).sum()} samples are assigned to more than three clusters, "
          f"{(cluster_cnt == 0).sum()} samples are not assigned to any cluster. "
          f"The smallest cluster has {cluster_label[train_index].sum(0).min()} samples, "
          f"the largest cluster has {cluster_label[train_index].sum(0).max()} samples.")
    return cluster_probs[train_index], cluster_label_prob, cluster_label_list
    