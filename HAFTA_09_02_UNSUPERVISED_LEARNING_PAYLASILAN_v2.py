################################
# Unsupervised Learning
################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)


################################
# K-Means
################################

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df.isnull().sum()
df.info()
df.describe().T

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]

kmeans = KMeans()
k_fit = kmeans.fit(df)

k_fit.get_params()

k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_
k_fit.inertia_

################################
# Kümelerin Görselleştirilmesi
################################

k_means = KMeans(n_clusters=2).fit(df)
kumeler = k_means.labels_
type(df)
df = pd.DataFrame(df)

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")
plt.show()

# merkezlerin isaretlenmesi
merkezler = k_means.cluster_centers_

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")

plt.scatter(merkezler[:, 0],
            merkezler[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

################################
# Optimum Küme Sayısının Belirlenmesi
################################

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

ssd

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

################################
# Final Cluster'ların Oluşturulması
################################


kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kumeler = kmeans.labels_

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

pd.DataFrame({"Eyaletler": df.index, "Kumeler": kumeler})

df["cluster_no"] = kumeler

df.head()

df["cluster_no"] = df["cluster_no"] + 1

df.groupby("cluster_no").agg({"cluster_no": "count"})

df[df["cluster_no"] == 6]

df.groupby("cluster_no").agg(np.mean)

################################
# Hierarchical Clustering
################################


df = pd.read_csv("datasets/USArrests.csv", index_col=0)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()

plt.figure(figsize=(15, 10))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dend = dendrogram(hc_average,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size=10)
plt.show()

dir(dend)



################################
# Principal Component Analysis
################################

df = pd.read_csv("datasets/Hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != "O"]
df = df[num_cols]

df.dropna(inplace=True)
df.shape

df = StandardScaler().fit_transform(df)

pca = PCA(5)
pca_fit = pca.fit_transform(df)


pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

################################
# Optimum Bileşen Sayısı
################################

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()


################################
# Final PCA'in Oluşturulması
################################


pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)
np.cumsum(pca.explained_variance_ratio_)
pca.explained_variance_ratio_

pca_fit.shape
df.shape









