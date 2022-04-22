import sys
import numpy as np
import numpy.random
import spkmeansmodule as kp
np.random.seed(0)


def euclidean_dist(vector1, vector2):
    dist = (vector1 - vector2) ** 2
    return np.sqrt(sum(dist))


def print_resault(resault, index, n):
    if index is not None:
        print(*index, sep=",")
    resault = ['%.4f' % resault[i] for i in range(len(resault))]
    resault = [resault[i:i + n] for i in range(0, len(resault), n)]
    for i in range(len(resault)):
        print(*(resault[i]), sep=",")


def kmeans_pp(vectors, k):
    index = []
    clusters = np.zeros((k, len(vectors[0])), dtype="float")
    first_cluster_index = np.random.choice(len(vectors))
    index.append(first_cluster_index)
    clusters[0] = np.copy(vectors[first_cluster_index])
    distances = np.full(len(vectors), sys.float_info.max)
    i = 1
    while i < k:
        for t in range(len(vectors)):  # compute distances from all vectors to the closest cluster
            for j in range(i):
                dist = euclidean_dist(vectors[t], clusters[j])
                if dist < distances[t]:
                    distances[t] = dist
        probabilities = distances / sum(distances)  # compute probabilities
        chosen_index = int(np.random.choice(len(vectors), 1, True, probabilities))
        clusters[i] = np.copy(vectors[chosen_index])
        index.append(chosen_index)
        i += 1
    return clusters, index


def k_legal(k):
    if k - int(k) > 0 or k < 0:
        print("Invalid Input!")
        return False
    return True


def execute_by_goal(vectors, goal):
    goal_to_function = {
        "wam": kp.wam,
        "ddg": kp.ddg,
        "lnorm": kp.lnorm,
        "jacobi": kp.jacobi,
    }
    return goal_to_function[goal](len(vectors), len(vectors[0]), vectors.flatten().tolist())


def create_vectors_from_file(file):
    try:
        vectors = np.genfromtxt(file, delimiter=",")
        return vectors
    except IOError:
        print("Invalid Input!")
        return None


def goal_is_spk(vectors, k):
    n = len(vectors[0])
    if not k_legal(k):
        return 1
    k = int(k)
    if k == 0:
        res = kp.spk(len(vectors), len(vectors[0]), vectors.flatten().tolist())
        if res is None:
            return None, None, None
        k = res[0]
        vectors = np.array(res[1]).reshape(int(len(res[1]) / k), k)
        n = int(k)

    init_clusters, chosen_index = kmeans_pp(vectors, k)
    clusters = kp.fit(k, len(vectors), len(vectors[0]), vectors.flatten().tolist(), init_clusters.flatten().tolist())
    return clusters, chosen_index, n


def main():
    cmd_input = sys.argv
    k = float(cmd_input[1])
    goal = cmd_input[2]
    file_name = cmd_input[3]

    vectors = create_vectors_from_file(file_name)
    if vectors is None:
        return 1

    chosen_index = None
    n = len(vectors)

    if goal == "spk":
        resualt, chosen_index, n = goal_is_spk(vectors, k)
    else:
        resualt = execute_by_goal(vectors, goal)
    if resualt is None:
        print("An Error Has Occurred")
        return 1

    print_resault(resualt, chosen_index, n)
    return 0


if __name__ == '__main__':
    main()
