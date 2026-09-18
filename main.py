import os
import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(42)


# ---------- 1. Génération de données synthétiques (3 blobs) ----------
def make_blobs(n_samples=500, centers=None, std=0.7, rng=None):
    if centers is None:
        centers = np.array([[2, 2], [-2, -1], [1, -3]])
    X = []
    for c in centers:
        X.append(rng.normal(loc=c, scale=std, size=(n_samples // len(centers), 2)))
    return np.vstack(X)


X = make_blobs(rng=rng)

# ---------- 2. Entraînement ----------
km = KMeans(n_clusters=3, init="kmeans++", random_state=0)
km.fit(X)

print(f"Itérations : {km.n_iter_}")
print(f"Inertie finale : {km.inertia_:.3f}")

# ---------- 3. Visualisation du clustering ----------
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#4C72B0", "#DD8452", "#55A868"]
for j in range(km.n_clusters):
    ax.scatter(X[km.labels_ == j, 0], X[km.labels_ == j, 1],
               s=20, color=colors[j], label=f"Cluster {j}")
ax.scatter(km.centroids[:, 0], km.centroids[:, 1],
           marker="X", s=250, c="red", edgecolor="black", label="Centroïdes")
ax.set_title("k-means from scratch — 3 clusters")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/clustering.png", dpi=150)
plt.close()

# ---------- 4. Courbe du coude ----------
inertias = []
ks = range(1, 11)
for k in ks:
    model = KMeans(n_clusters=k, random_state=0).fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(ks), inertias, marker="o")
plt.xlabel("Nombre de clusters k")
plt.ylabel("Inertie intra-cluster")
plt.title("Méthode du coude")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/elbow.png", dpi=150)
plt.close()

# ---------- 5. Convergence de l'inertie ----------
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(km.history_) + 1), km.history_, marker="o")
plt.xlabel("Itération")
plt.ylabel("Inertie")
plt.title("Convergence de k-means")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/convergence.png", dpi=150)
plt.close()

# ---------- 6. Comparaison avec sklearn (validation uniquement) ----------
try:
    from sklearn.cluster import KMeans as SKKMeans
    sk = SKKMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    print(f"\nInertie sklearn : {sk.inertia_:.3f}")
    print(f"Inertie maison  : {km.inertia_:.3f}")
except ImportError:
    print("sklearn non installé — comparaison ignorée.")

print("\nFigures sauvegardées dans figures/")