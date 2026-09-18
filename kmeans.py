"""
Implémentation from scratch de l'algorithme des k-moyennes (k-means).
Aucune dépendance à sklearn.cluster.
"""
import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-6,
                 init="kmeans++", random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.history_ = []  # inertie à chaque itération

    # ---------- Initialisation ----------
    def _init_forgy(self, X, rng):
        idx = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[idx].copy()

    def _init_kmeanspp(self, X, rng):
        n_samples, _ = X.shape
        centroids = [X[rng.integers(n_samples)]]
        for _ in range(1, self.n_clusters):
            # distance au carré de chaque point au centroïde le plus proche
            d2 = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = d2 / d2.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])
        return np.array(centroids)

    # ---------- Étapes de l'algo ----------
    @staticmethod
    def _assign(X, centroids):
        # distances euclidiennes au carré : (n, k)
        d2 = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        return np.argmin(d2, axis=1)

    @staticmethod
    def _update(X, labels, k):
        centroids = np.zeros((k, X.shape[1]))
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
            else:
                # cluster vide : on le laisse tel quel (ou on réinitialise)
                centroids[j] = X[np.random.randint(X.shape[0])]
        return centroids

    @staticmethod
    def _inertia(X, labels, centroids):
        return float(np.sum((X - centroids[labels]) ** 2))

    # ---------- API ----------
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        if self.init == "kmeans++":
            self.centroids = self._init_kmeanspp(X, rng)
        else:
            self.centroids = self._init_forgy(X, rng)

        for it in range(self.max_iter):
            labels = self._assign(X, self.centroids)
            new_centroids = self._update(X, labels, self.n_clusters)

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.labels_ = labels
            self.inertia_ = self._inertia(X, labels, self.centroids)
            self.history_.append(self.inertia_)
            self.n_iter_ = it + 1

            if shift < self.tol:
                break

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._assign(X, self.centroids)