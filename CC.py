# %% import
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import time
from plotnine import (
    ggplot, aes,
    geom_path,
    coord_equal,
    scale_color_brewer,
    theme,
    element_text, element_blank, element_rect,
    geom_point,
    labs
)

from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import MinMaxScaler

#%% ADMM update

# update U
def update_U(X, V, Lambda, D, nu):

    n, p = X.shape

    # (rho D_row^T D_row + pi_X I + rho I)
    A = np.eye(n) + nu * (D.T @ D)

    # (pi_X X + rho D_row^T (V_row - Q_row) + rho (M^T - N^T))
    B = X + nu * (D.T @ (V - Lambda/nu))

    U = np.linalg.inv(A) @ B

    return U

# Analytical Solution of prox
def prox(x, tau):

    x = np.asarray(x)
    lv = np.linalg.norm(x, axis=1)

    s = np.maximum(0.0, 1.0 - tau / lv)


    return s[:, None] * x


# update V
def update_V(U, Lambda, w, gamma, nu, D, DU):

    q, p = U.shape
    Z = DU + Lambda/nu
    tau = (gamma / nu) * w            
    V = prox(Z, tau)

    return V

# Update Lambda
def update_Lambda(Lambda, V, nu, DU):

    # U[:, i] - U[:, j]
    Z = DU - V

    return Lambda + nu * Z

# %% build weight
#"Thin" a weight vector to be positive only for its k-nearest neighbors
#return A vector of weights for convex clustering.


# KNN graph for weights
def build_knn_graph(X, k):

    X = np.asarray(X, float)
    n = X.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Remove self-neighbor
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    edges_set = set()
    edge_dist_list = []

    for i in range(n):
        for j_idx, dist in zip(indices[i], distances[i]):
            # constraint i < j
            a, b = sorted([i, j_idx])
            if (a, b) not in edges_set:
                edges_set.add((a, b))
                edge_dist_list.append(dist)

    edges = np.array(list(edges_set), dtype=int)

    return edges


def incidence_matrix(n, edges):

    edges = np.asarray(edges, int)
    K = edges.shape[0]
    D = np.zeros((K, n), dtype=float)
    a = edges[:, 0]
    b = edges[:, 1]
    D[np.arange(K), a] = 1.0
    D[np.arange(K), b] = -1.0
    return D

def compute_weights(X, edges, phi):

    X = np.asarray(X, float)

    dist2 = np.array(
        [np.sum((X[i] - X[j]) ** 2) for (i, j) in edges],
        dtype=float,
    )
    # Gaussian kernel weights
    w = np.exp(-phi * dist2)

    return w


# %% ADMM

#Convex Clustering via ADMM
#n is the number of data points
#p is the number of features
#m is the number non-zero weights

# ADMM stopping rule
def residual(
    DU,
    V,
    V_prev,
    Lam,
    Dt,
    nu,
    abs_tol,
    rel_tol,
):

    # primal residual
    R_pri = DU - V
    r_norm = np.linalg.norm(R_pri)

    # dual residual
    s_norm = np.linalg.norm(nu * (Dt @ (V - V_prev)))

    # problem dimensions
    m, p = V.shape
    n = Dt.shape[0]

    sqrt_mp = np.sqrt(m * p)
    sqrt_np = np.sqrt(n * p)

    # stopping thresholds (Boyd et al.)
    eps_pri = abs_tol * sqrt_mp + rel_tol * max(
        np.linalg.norm(DU), np.linalg.norm(V)
    )
    eps_dual = abs_tol * sqrt_np + rel_tol * np.linalg.norm(
        nu * (Dt @ Lam)
    )

    return r_norm, s_norm, eps_pri, eps_dual



def admm(
    X, Lambda, V, D, w, gamma,
    nu=1.0, max_iter=100, tol=1e-4,
    save_path=None
):

    X = np.asarray(X, dtype=np.float64)
    Lambda = np.asarray(Lambda, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    gamma = float(gamma)
    nu = float(nu)
    max_iter = int(max_iter)
    tol = float(tol)

    q, p = X.shape

    U = np.zeros((q, p), dtype=np.float64)

    for it in range(max_iter):

        # U step
        U = update_U(X, V, Lambda, D, nu)

        # V step
        V_old = V.copy()
        DU = D @ U
        V = update_V(U, Lambda, w, gamma, nu, D, DU)

        # Lambda step
        Lambda = update_Lambda(Lambda, V, nu, DU)

        # residual
        r_norm, s_norm, eps_pri, eps_dual = residual(
            DU=DU,
            V=V,
            V_prev=V_old,
            Lam=Lambda,
            Dt=D.T,
            nu=nu,
            abs_tol=tol,
            rel_tol=tol,
        )


        # stopping rule
        if (r_norm <= eps_pri) and (s_norm <= eps_dual):
            print(f"gamma={gamma:.2f}, converged at iter {it}.")
            break


    return U, V, Lambda


# %% Find clusters

#uses breadth-first search to identify the connected components of the corresponding
#adjacency graph of the centroid differences vectors


def find_clusters(A):

    graph = csr_matrix(A)

    n_clusters, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    _, size = np.unique(labels, return_counts=True)

    return labels, size



def get_clustering_labels(U, tol=1e-4):

    row_diff = U[:, None, :] - U[None, :, :]
    row_dists = np.linalg.norm(row_diff, axis=2)
    
    # Build adjacency matrix:
    # nodes are connected if distance < tolerance
    A_row = (row_dists < tol).astype(int)

    labels, sizes = find_clusters(A_row)

    return labels

# %% plot single gamma result

def plot_X_and_U(
    X_orig,
    labels_full,
    U_cluster,
    title="PCA"
):

    X_orig = np.asarray(X_orig, float)
    labels_full = np.asarray(labels_full, int)
    U_cluster = np.asarray(U_cluster, float)

    # fixed PCA transform
    pca = PCA(n_components=2, random_state=42)
    Z_orig = pca.fit_transform(X_orig)
    Zc = pca.transform(U_cluster)
    uniq = np.unique(labels_full)
    K = len(uniq)
    cmap = plt.colormaps.get_cmap("tab10").resampled(K)
    plt.figure(figsize=(6.5, 6))

    # plot X
    for i, k in enumerate(uniq):
        mask = (labels_full == k)
        plt.scatter(
            Z_orig[mask, 0], Z_orig[mask, 1],
            s=28, edgecolors='black', linewidths=0.3,
            c=[cmap(i)],
            label=None
        )

    # plot U
    plt.scatter(
        Zc[:, 0], Zc[:, 1],
        s=40,
        marker='o',
        facecolors='none',
        edgecolors='red',
        linewidths=1.2,
        label="Cluster centers"
    )

    plt.title(f"{title} | Clusters={K}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% clustering path plotting

# ----------------------------
# project U and X
# ----------------------------
def project_all_to_pca_2d(X_row, U_path):
    """
    X_row : ndarray, shape (n, p)
    U_path : list of ndarray, each shape (n, p)

    Returns
    -------
    X_pca : ndarray, shape (n, 2)
    U_path_pca : list of ndarray, each shape (n, 2)
    pca : fitted PCA object
    """
    all_data = [X_row] + U_path
    stacked = np.vstack(all_data)   # ((T+1)*n, p)

    pca = PCA(n_components=2)
    stacked_pca = pca.fit_transform(stacked)

    n = X_row.shape[0]
    T = len(U_path)

    X_pca = stacked_pca[:n]
    U_path_pca = []

    start = n
    for _ in range(T):
        U_path_pca.append(stacked_pca[start:start+n])
        start += n

    return X_pca, U_path_pca, pca


# ----------------------------
# 2) build path dataframe
# ----------------------------
def build_tree_df_2d(U_path_2d):
    rows = []
    for step, U_row in enumerate(U_path_2d):
        n, d = U_row.shape
        for i in range(n):
            rows.append({
                "id": int(i),
                "step": int(step),
                "x1": float(U_row[i, 0]),
                "x2": float(U_row[i, 1]),
            })
    return pd.DataFrame(rows)


# ----------------------------
# 3) compute centers by labels
# ----------------------------
def centers_from_labels(U, labels):
    labels = np.asarray(labels).astype(int)
    uniq = np.unique(labels)

    centers = []
    for k in uniq:
        mask = (labels == k)
        centers.append(U[mask].mean(axis=0))
    centers = np.vstack(centers)

    return uniq, centers


# ----------------------------
# 4) find first index where K == target_K
# ----------------------------
def find_first_k_index(K_list, target_K=3):
    for i, k in enumerate(K_list):
        if int(k) == int(target_K):
            return i
    raise ValueError(f"K={target_K} not found in K_list.")


# ----------------------------
# 5) main plotting function (PCA version)
# ----------------------------
def plot_recovery_tree_in_pca(
    X_row,
    U_path,
    labels_path,
    K_list,
    true_labels=None,
   # target_center_k=3,
    path_alpha=0.65,
    path_size=0.5,
    point_size=2.8,
    center_size=5.0
):


    # ----------------------------
    # PCA projection
    # ----------------------------
    X_pca, U_path_pca, pca = project_all_to_pca_2d(X_row, U_path)

    # ----------------------------
    # path dataframe
    # ----------------------------
    df_paths = build_tree_df_2d(U_path_pca).sort_values(["id", "step"])

    # ----------------------------
    # point colors
    # ----------------------------
    if true_labels is None:
        base_labels = np.asarray(labels_path[0]).astype(int)
    else:
        base_labels = np.asarray(true_labels).astype(int)

    df_points = pd.DataFrame({
        "x1": X_pca[:, 0],
        "x2": X_pca[:, 1],
        "cluster_true": base_labels
    })


    all_x = np.concatenate([
        df_paths["x1"].to_numpy(),
        df_points["x1"].to_numpy()
    ])
    all_y = np.concatenate([
        df_paths["x2"].to_numpy(),
        df_points["x2"].to_numpy()
    ])

    x_margin = 0.08 * (all_x.max() - all_x.min() + 1e-12)
    y_margin = 0.08 * (all_y.max() - all_y.min() + 1e-12)

    xlim = (all_x.min() - x_margin, all_x.max() + x_margin)
    ylim = (all_y.min() - y_margin, all_y.max() + y_margin)

    # ----------------------------
    # plot
    # ----------------------------
    p = (
        ggplot()

        # path
        + geom_path(
            data=df_paths,
            mapping=aes(x="x1", y="x2", group="id"),
            size=path_size,
            alpha=path_alpha,
            color="gray"
        )

        # original points with true-label colors
        + geom_point(
            data=df_points,
            mapping=aes(x="x1", y="x2", color="factor(cluster_true)"),
            size=point_size
        )

        + scale_color_brewer(type="qual", palette="Set1")


        + coord_equal(xlim=xlim, ylim=ylim)

        + labs(x="PC1", y="PC2")

        + theme(
            legend_position="none",
            panel_background=element_rect(fill="white", color=None),
            plot_background=element_rect(fill="white", color=None),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            panel_border=element_rect(color="black", fill=None, size=1.0),
            axis_title=element_text(size=12),
            axis_text=element_text(size=10)
        )
    )

    return p


# %% Conduct Convex clustering


#gamma sequence
def make_gamma_sequence(gamma_min=1e-2, gamma_max=50.0, n_gamma=20):
    return np.logspace(np.log10(gamma_min),
                       np.log10(gamma_max),
                       num=n_gamma)


def run_single_gamma(X, D, w, gamma=10.0, nu=1.0, max_iter=200, tol=1e-4,):
    X = np.asarray(X, float)
    n, d = X.shape

    K = D.shape[0]            
    V0 = np.zeros((K, d))      
    L0 = np.zeros_like(V0)     

    U, V, Lambda = admm(X, L0, V0, D, w, gamma=gamma, nu=nu, max_iter=max_iter, tol=tol,save_path = r"E:\code\admm_logs")


    labels = get_clustering_labels(U, tol=1e-2)

    return U, V, Lambda, labels


def run_gamma_sequence(
    X,
    D,
    w,
    gamma_seq,
    nu=1.0,
    max_iter=2000,
    tol=1e-4,
    plot_each_gamma=True,
):
   
    # 

    X = np.asarray(X, float)
    n0, d = X.shape

    U_path = []
    labels_path = []
    K_list = []

    for g in gamma_seq:

        # initial location of U
        if g == 0.0:
            U0 = X.copy()
            labels0 = np.arange(n0, dtype=int)

            U_path.append(U0)
            labels_path.append(labels0)
            K_list.append(n0)

            if plot_each_gamma:
                plot_X_and_U(
                    X,
                    labels0,
                    U0,
                    title=f"γ={g}"
                )
            continue

        # ADMM
        U, V, Lambda, labels = run_single_gamma(
            X,
            D,
            w,
            gamma=g,
            nu=nu,
            max_iter=max_iter,
            tol=tol,
        )

 
        K_curr = int(labels.max() + 1)

        U_full = U         # (n0, d)
        labels_full = labels     # (n0,)
        K_curr = int(labels_full.max() + 1)

        U_path.append(U_full)
        labels_path.append(labels_full)
        K_list.append(K_curr)


        if plot_each_gamma:
            plot_X_and_U(
                X,
                labels,
                U,
                title=f"γ={g:.2f}"
            )

    return U_path, labels_path, K_list


# %% Data Generation

np.random.seed(100540)

# =========================
# data setting
# =========================
num_clusters = 3
points_per_cluster = 4
dim = 2

# =========================
# Randomly generate group centers
# =========================
centers = np.random.randn(num_clusters, dim) 

# =========================
# sampling from gaussian distribution
# =========================
noises = 0.3

X_list = []
y_list = []

for k, c in enumerate(centers):
    cluster = c + np.random.normal(
    loc=0.0,
    scale=noises,
    size=(points_per_cluster, dim)
    )
    
    X_list.append(cluster)
    y_list.append(np.full(points_per_cluster, k))
    
    X = np.vstack(X_list)
    y_true = np.concatenate(y_list)


# Standardization
scaler = MinMaxScaler(feature_range=(-1, 1))
X= scaler.fit_transform(X)
X = np.round(X, 4)

print("X shape:", X.shape) # (25, 100)

# =========================
# PCA
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)



# =========================
# Visualization
# =========================

plt.figure(figsize=(6, 6))

for k in range(num_clusters):
    mask = y_true == k
    plt.scatter(
    X_pca[mask, 0],
    X_pca[mask, 1],
    s=40,
    label=f"Cluster {k+1}"
    )

plt.title("Data Visualization(PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.show()


# %% parameter settings

cfg = {
    # gamma path
    "gamma_min": 0.2,
    "gamma_max": 3,
    "n_gamma": 20,

    # number of knn / weights
    "k": 4,
    "phi": 0.25,

    # ADMM
    "nu": 0.1,
    "max_iter": 5000,
    "tol": 1e-3,

    # plot
    "plot_each_gamma": True,
}



# %% main

# main
if __name__ == "__main__":

    # gamma sequence
    gamma_seq = np.concatenate([
        np.array([0.0]),
        make_gamma_sequence(
            gamma_min=cfg["gamma_min"],
            gamma_max=cfg["gamma_max"],
            n_gamma=cfg["n_gamma"]
        )
    ])
    # graph and weights
    n, p = X.shape

    edges = build_knn_graph(
        X,
        k=cfg["k"]
    )

    D = incidence_matrix(
        n,
        edges
    )
    
    # weight 
    w = compute_weights(
        X,
        edges,
        phi=cfg["phi"]
    )

    print("weight information:")
    print("w_min,w_max:", w.min(), w.max())
    print("w_mean,std:", w.mean(), w.std())
    q = np.quantile(w, [0.01, 0.1, 0.5, 0.9, 0.99])
    print("quantiles 1%,10%,50%,90%,99%:\n", q)

    # run gamma path
    U_path, labels_path, K_list = run_gamma_sequence(
        X=X,
        D=D,
        w=w,
        gamma_seq=gamma_seq,
        nu=cfg["nu"],
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
        plot_each_gamma=cfg["plot_each_gamma"]
    )
    # ============================
    # plot clustering path
    # ============================
    p = plot_recovery_tree_in_pca(
        X_row=X,
        U_path=U_path,
        labels_path=labels_path,
        K_list=K_list,
        true_labels=y_true
    )
    p

    
# %%
#plot
p