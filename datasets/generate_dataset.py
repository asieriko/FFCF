import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs

# Define plot functions:
def plot_client(X, y, colors, marker, title, show=False):
    plt.figure(1)

    for k, col in enumerate(colors):
        cluster_data = y == k
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=marker, s=10)

    plt.title(title)
    plt.ylim(-5, 15)
    plt.xlim(-5,15)
    if show:
        plt.show()
    plt.savefig(title+".png")

def plot_data(Xv, yv, markers, title, show=False):
    # Plot init seeds along side sample data
    plt.figure()

    for k, col in enumerate(colors):
        for X,y in zip(Xv, yv):
            cluster_data = y == k
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker="+", s=10)
            plt.ylim(-5, 15)
            plt.xlim(-5,15)

    plt.title("Clusters All Clients")
    if show:
        plt.show()
    plt.savefig(title+"_v.png")

    # 3 clients in same row
    fig, axiss = plt.subplots(1, len(Xv))
    for ax in axiss:
        ax.set_xlim((-5, 15))
        ax.set_ylim((-5, 15))
    fig.set_size_inches(20, 6)
    # fig.suptitle('Data distribution among the 3 clients')
    for X,y,m,ax in zip(Xv, yv, markers,axiss):
        for k, col in enumerate(colors):
            cluster_data = y == k
            ax.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=m, s=10)
            ax.set_title("Client 1")
    if show:
        plt.show()
    plt.savefig(title+"_h.png")

# The clusters are generated by sampling from Gaussian distributions centered at 
# μ1 = (0, 0), μ2 = (0, 10), μ3 = (10, 10), and μ4 = (10, 0) with standard deviation of 1.0. 
# Client 1 has data drawn from Gaussians with μ1, μ2, c
# lient 2 μ2, μ3 and 
# client 3 μ3, μ4.

# Generate sample data
centers1 = [[0,0],[0,10],[10,10],[10,0]]

X1, y_true1 = make_blobs(
    n_samples=[500,500,0,0], centers=centers1,random_state=0
)

X2, y_true2 = make_blobs(
    n_samples=[0,50,50,0], centers=centers1, random_state=0
)

X3, y_true3 = make_blobs(
    n_samples=[0,0,50,50], centers=centers1,random_state=0
)

# Save data
d = []
for i,(x,y) in enumerate(zip([X1,X2,X3],[y_true1,y_true2,y_true3])):
    c = np.full(y.shape,f"c{i}")
    di = np.column_stack((c,x,y))
    d.extend(di)

np.savetxt('C3C4Lin3.csv', d, fmt="%s",delimiter=",")  

# Plot data
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
markers = [".","X","+"]

plot_client(X1, y_true1, colors, ".", "Points for Client 1")
plot_client(X2, y_true2, colors, "X", "Points for Client 2")
plot_client(X3, y_true3, colors, "+", "Points for Client 3")

plot_data([X1,X2,X3],[y_true1, y_true2, y_true3],markers,"Points for all data")



