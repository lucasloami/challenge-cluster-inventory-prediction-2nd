from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
import matplotlib.pyplot as plt

# HELPER FUNCTION DEFINITIONS
def plot_elbow_curve(X, K):
    # scipy.cluster.vq.kmeans
    KM = [kmeans(X,k) for k in K]
    centroids = [cent for (cent,var) in KM]   # cluster centroids
    avgWithinSS = [var for (cent,var) in KM] # mean within-cluster sum of squares

    # alternative: scipy.cluster.vq.vq
    #Z = [vq(X,cent) for cent in centroids]
    #avgWithinSS = [sum(dist)/X.shape[0] for (cIdx,dist) in Z]

    # alternative: scipy.spatial.distance.cdist
#     D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
#     cIdx = [np.argmin(D,axis=1) for D in D_k]
#     dist = [np.min(D,axis=1) for D in D_k]
#     avgWithinSS = [sum(d)/X.shape[0] for d in dist]

    ##### plot ###
    kIdx = 3
    # elbow curve
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
        markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()

