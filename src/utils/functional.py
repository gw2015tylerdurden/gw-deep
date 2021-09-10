import torch
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import colorsys
from collections import abc
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage
import re


def get_mean_std(loader):
    channels_sum, channels_squared_sum, n = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        n += 1
    mean = channels_sum / n
    std = (channels_squared_sum / n - mean ** 2) ** 0.5
    return mean, std


def segmented_cmap(cmap, num_split=10):
    cmap = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=0, vmax=cmap.N)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    tmp = [mapper.to_rgba(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", tmp, N=num_split)
    return cmap


def darken(c, amount=0.5):
    c = colorsys.rgb_to_hls(*colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def acronym(name):
    name = re.sub(
        r"(^[0-9a-zA-Z]{5,}(?=_))|((?<=_)[0-9a-zA-Z]*)",
        lambda m: str(m.group(1) or "")[:3] + str(m.group(2) or "")[:1],
        name,
    )
    name = name.replace("_", ".")
    return name


def seriation(Z, N, cur_index):
    """
    input:
        - Z is a hierarchical tree (dendrogram)
        - N is the number of points given to the clustering process
        - cur_index is the position in the tree for the recursive traversal
    output:
        - order implied by the hierarchical tree Z

    seriation computes the order implied by a hierarchical tree (dendrogram)
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(X, method="ward"):
    """
    input:
        - dist_mat is a distance matrix
        - method = ["ward","single","average","complete"]
    output:
        - seriated_dist is the input dist_mat,
          but with re-ordered rows and columns
          according to the seriation, i.e. the
          order implied by the hierarchical tree
        - res_order is the order implied by
          the hierarhical tree
        - res_linkage is the hierarhical tree (dendrogram)

    compute_serial_matrix transforms a distance matrix into
    a sorted distance matrix according to the order implied
    by the hierarchical tree (dendrogram)
    """
    dist_mat = squareform(pdist(X))
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage

def cosine_similarity(x):
    x = x / x.norm(dim=-1)[:, None]
    return torch.mm(x, x.transpose(0, 1))

def sample_from_each_class(y, num_samples=10, random_seed=42):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced = {}
    for level, gb_idx in groupby_levels.items():
        indices = np.random.choice(gb_idx, size=num_samples, replace=True).tolist()
        balanced[level] = indices
    return balanced
