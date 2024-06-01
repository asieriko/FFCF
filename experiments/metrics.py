from sklearn.metrics import silhouette_score, adjusted_rand_score, f1_score
import numpy as np
from FFCF.fcf import euclidean_distance

def offline_stats(fcfsc, Vmm, X_test, y_test):
    """
    Computes stats for the offline step given the summarizer structure and the last chunk of data

    Args:
        fcfsc: fcfs centers
        Vmm: Membership to each center
        chunk: Pandas Data frame. Each column is an attribute and the last column should be the class

    Returns:
        ari, sil: Adjusted Rand Index, Silhouette score
    """
    clusters = np.argmax(Vmm, axis=0)
    clusters[np.max(Vmm, axis=0) < 0.5] = -1  # if the highest membership to a cluster is < 0.5 then it is an outlier
    point_fcf = []
    for xi in X_test:
        d_min = euclidean_distance(xi, fcfsc[0])
        id_min = 0
        for i, fm in enumerate(fcfsc):
            d = euclidean_distance(xi, fm)
            if d < d_min:
                d_min = d
                id_min = i
        point_fcf.append(clusters[id_min])
    y = y_test

    y_h = np.array(point_fcf)
    # not_nans = np.where(y.astype(str) != 'nan')[0].astype(int)
    nans = np.where(y.astype(str) == 'nan')[0].astype(int)
    y[nans] = -1
    y = y.astype(int)
    # ari = adjusted_rand_score(y[not_nans], y_h[not_nans])
    ari = adjusted_rand_score(y, y_h)
    # f1 = f1_score(y, y_h)
    if len(np.unique(y_h))==1:
        sil = 0
    else:
        sil = silhouette_score(X_test, y_h)

    return ari, sil

def squared_errors(test_prediction, test_data, macro_centers):
    test_prediction = np.array(test_prediction)
    test_points = test_data[0]
    WSSE = 0
    unique_clusters = np.unique(test_prediction)
    for cluster in unique_clusters:
        idx_cluster = np.where(test_prediction==cluster)[0]
        for idx in idx_cluster:
            WSSE += euclidean_distance(test_points[idx],macro_centers[cluster])
    WSSE = WSSE / (len(test_points)*len(test_points[0]))

    OSSE = 0
    unique_clusters = np.unique(test_prediction)
    for cluster in unique_clusters:
        idx_cluster = np.where(test_prediction!=cluster)[0]
        for idx in idx_cluster:
            OSSE += euclidean_distance(test_points[idx],macro_centers[cluster])
    OSSE = OSSE / (len(test_points)*len(test_points[0]))

    print(f"{WSSE=}")
    print(f"{OSSE=}")
    print(macro_centers)
