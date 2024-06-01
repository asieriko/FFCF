import pandas as pd
import numpy as np

import sys, os
sys.path.append(os.path.abspath("."))
from FFCF.global_fcf_agent import GlobalFCFAgent
from FFCF.fcf import euclidean_distance
from metrics import offline_stats, squared_errors


def load_towards(fpath="datasets/C3C4Linear.csv"):
    df = pd.read_csv(fpath,names=["Client","X","Y","class"])
    C0 = df[df.Client == "c0"]
    C1 = df[df.Client == "c1"]
    C2 = df[df.Client == "c2"]
    C0 = [C0[["X","Y"]],C0["class"].to_numpy()]
    C1 = [C1[["X","Y"]],C1["class"].to_numpy()]
    C2 = [C2[["X","Y"]],C2["class"].to_numpy()]
    return [C0,C1,C2]


for file_path in ["datasets/C3C4Lin1.csv", "datasets/C3C4Lin2.csv", "datasets/C3C4Lin3.csv","datasets/C3C4Lin.csv"]:
    
    # Algorithm
    clients_datasets = load_towards(file_path)
    fedmic = GlobalFCFAgent(n_clusters=4)
    fedmic.init_clients(clients_datasets)
    fedmic.train_clients()
    fedmic.update_clients()
    macro_centers, center_memeberhisp, centers = fedmic.generate_final_clusters()

    # Client Evaluation
    print(f"Outliers count: {sum(np.array([max(center_memeberhisp[:,i]) for i in range(len(center_memeberhisp[0]))]) < 0.5)}")
    for client in clients_datasets:
        X = client[0].values
        y = client[1]
        ari, sil = offline_stats(centers, center_memeberhisp, X, y)
        print(f"{ari=},{sil=}")

    # Ground Truth Gap
    obj = [[10,0],[0,10],[0,0],[10,10]]
    dist = lambda x,y: np.sqrt(sum((x-y)**2))
    gap = []
    for i in range(len(obj)):
        gap.append(dist(obj[i],macro_centers[i]))

    # Corresponding clusters
    df = pd.read_csv(file_path,names=["Client","X","Y","class"])
    test_data = [df[["X","Y"]].to_numpy(),df["class"].to_numpy()]
    test_prediction = []
    for point in test_data[0]:
        min_d = dist(point, macro_centers[0])
        min_idx = 0
        for i, mc in enumerate(macro_centers):
            d = dist(point, mc)
            if d < min_d:
                min_d = d
                min_idx = i
        test_prediction.append(min_idx)

    # Evaluation
    print(file_path)
    squared_errors(test_prediction, test_data, macro_centers)