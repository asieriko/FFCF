from math import sqrt
from sklearn.model_selection import train_test_split
import numpy as np
from .fcf import FCF, merge_fcfs, euclidean_distance
from .WFCM import WFCM


class LocalFCFAgent:

    def __init__(self, min_fcfs=5, max_fcfs=100, merge_threshold=1.0, radius_factor=1.0, m=2.0, n_clusters=3):
        self.min_fcfs = min_fcfs
        self.max_fcfs = max_fcfs
        self.merge_threshold = merge_threshold
        self.radius_factor = radius_factor
        self.m = m
        self.__fcfs = []
        self.__global_fcfs = []
        self.centers = None
        self.data = []
        self.n_macro_clusters = n_clusters

    def set_data(self, data):
        self.data = data
        self.X = data[0].to_numpy()
        self.y = data[1]
        # Split the data into a training set and a test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33,
                                                                                random_state=0)

    def predict(self, X):
        X = self.X_test
        if self.centers:
            clusters = [np.argmin([euclidean_distance(x,c) for c in self.centers]) for x in X]
            return clusters
        else:
            print("No clusters...")

    def fit(self):
        for x in self.X_train:
            self.summarize(x)
        self._clean_fcfs()
        return self.__fcfs

    def generate_final_clusters(self):
        centers = np.array([fm.center.tolist() for fm in self.__fcfs])
        weights = [fm.m for fm in self.__fcfs]  # Sum of membership
        macro_centers, center_memeberhisp = WFCM(centers, weights, c=self.n_macro_clusters)
        return macro_centers, center_memeberhisp

    def update(self, new_fcfs):
        self.__global_fcfs = new_fcfs

    def summarize(self, values):
        if len(self.__fcfs) < self.min_fcfs:
            self.__fcfs.append(FCF(values))
            return
        
        distance_from_fcfs = [euclidean_distance(fcf.center, values) for fcf in self.__fcfs]
        is_outlier = True

        for idx, fcf in enumerate(self.__fcfs):
            if fcf.radius == 0.0:
                # Minimum distance from another fcf
                radius = min([
                    euclidean_distance(fcf.center, another_fcf.center)
                    for another_idx, another_fcf in enumerate(self.__fcfs)
                    if another_idx != idx
                ])
            else:
                radius = fcf.radius * self.radius_factor
            
            if distance_from_fcfs[idx] <= radius:
                is_outlier = False
        
        if is_outlier:
            if len(self.__fcfs) >= self.max_fcfs:
                oldest = min(self.__fcfs, key=lambda f: f.m)
                self.__fcfs.remove(oldest)
            self.__fcfs.append(FCF(values))
        else:
            memberships = self.__memberships(distance_from_fcfs)
            for idx, fcf in enumerate(self.__fcfs):
                fcf.assign(values, memberships[idx], distance_from_fcfs[idx])

        self.__fcfs = merge_fcfs(self.__fcfs, self.merge_threshold)

    def _clean_fcfs(self):
        while len(self.__fcfs) >= self.max_fcfs:
            smallest = min(self.__fcfs, key=lambda f: f.m)
            self.__fcfs.remove(smallest)

    def summary(self):
        return self.__fcfs.copy()

    def __memberships(self, distances):
        memberships = []
        for distance_j in distances:
            # To avoid division by 0
            sum_of_distances = 2.2250738585072014e-308
            for distance_k in distances:
                if distance_k != 0:
                    sum_of_distances += pow((distance_j / distance_k), 2. / (self.m - 1.))
            memberships.append(1.0 / sum_of_distances)
        return memberships
