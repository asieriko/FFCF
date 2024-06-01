import numpy as np
from .local_fcf_agent import LocalFCFAgent, merge_fcfs
from .fcf import FCF
from .WFCM import WFCM
from math import sqrt


class GlobalFCFAgent:

    def __init__(self, n_clients=3, n_clusters=3, min_fcfs=5, max_fcfs=100, merge_threshold=1.0, m=2.0):
        self.n_clients = n_clients
        self.min_fcfs = min_fcfs
        self.max_fcfs = max_fcfs
        self.merge_threshold = merge_threshold
        self.m = m
        self.n_macro_clusters = n_clusters
        self.__fcfs = []
        self.clients = {}

    def init_clients(self, client_data):
        for idx, data in enumerate(client_data):
            local_client = LocalFCFAgent(n_clusters=self.n_macro_clusters)
            local_client.set_data(data)
            self.clients[idx] = local_client

    def train_clients(self):
        for client in self.clients.values():
            self.append_fcfs(client.fit())
        self.__fcfs = merge_fcfs(self.__fcfs, self.merge_threshold)

    def update_clients(self):
        for client in self.clients.values():
            client.update(self.__fcfs)

    def append_fcfs(self, new_fcfs):
        self.__fcfs.extend(new_fcfs)

    def generate_final_clusters(self):
        fcf_centers = np.array([fm.center.tolist() for fm in self.__fcfs])
        weights = [fm.m for fm in self.__fcfs]  # Sum of membership
        macro_centers, center_memeberhisp = WFCM(fcf_centers, weights, c=self.n_macro_clusters)
        return macro_centers, center_memeberhisp, fcf_centers