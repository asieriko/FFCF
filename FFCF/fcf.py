from math import sqrt


class FCF:

    def __init__(self, cf):
        self.cf = cf.copy()
        self.m = 1.0
        self.n = 1
        self.ssd = 0.0
        self.center = cf.copy()
        self.radius = 0.0

    def assign(self, values, membership, distance):
        self.m += membership
        self.n += 1
        self.ssd += membership * pow(distance, 2)

        for idx, value in enumerate(values):
            self.cf[idx] += value * membership

        self.__update_center()
        self.__update_radius()

    def merge(fcf_a, fcf_b):
        merged_fcf = FCF([])
        for idx, cf_a in enumerate(fcf_a.cf):
            merged_fcf.cf.append(cf_a + fcf_b.cf[idx])
        merged_fcf.m = fcf_a.m + fcf_b.m
        merged_fcf.ssd = fcf_a.ssd + fcf_b.ssd
        merged_fcf.n = fcf_a.n + fcf_b.n
        merged_fcf.center = fcf_a.center.copy()

        merged_fcf.__update_center()
        merged_fcf.__update_radius()

        return merged_fcf

    def __update_center(self):
        for idx, cf_i in enumerate(self.cf):
            self.center[idx] = cf_i / self.m

    def __update_radius(self):
        self.radius = sqrt(self.ssd / self.n)


def euclidean_distance(value_a, value_b):
    sum_of_distances = 0
    for idx, value in enumerate(value_a):
        sum_of_distances += pow(value - value_b[idx], 2)
    return sqrt(sum_of_distances)

def merge_fcfs(fcfs, merge_threshold):
    fcfs_to_merge = []

    for i in range(0, len(fcfs) - 1):
        for j in range(i + 1, len(fcfs)):
            dissimilarity = euclidean_distance(fcfs[i].center, fcfs[j].center)
            dissimilarity = sqrt(sum([pow(a-b,2) for a,b in zip(fcfs[i].center,fcfs[j].center)]))
            sum_of_radius = fcfs[i].radius + fcfs[j].radius

            if dissimilarity != 0:
                similarity = sum_of_radius / dissimilarity
            else:
                # Highest value possible
                similarity = 1.7976931348623157e+308

            if similarity >= merge_threshold:
                fcfs_to_merge.append([i, j, similarity])

    # Sort by most similar
    fcfs_to_merge.sort(reverse=True, key=lambda k: k[2])
    merged_fcfs_idx = []
    merged_fcfs = []

    for (i, j, _) in fcfs_to_merge:
        if i not in merged_fcfs_idx and j not in merged_fcfs_idx:
            merged_fcfs.append(FCF.merge(fcfs[i], fcfs[j]))
            merged_fcfs_idx.append(i)
            merged_fcfs_idx.append(j)

    merged_fcfs_idx.sort(reverse=True)
    for idx in merged_fcfs_idx:
        fcfs.pop(idx)
    
    return fcfs + merged_fcfs
