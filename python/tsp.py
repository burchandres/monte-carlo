import numpy as np
import itertools
import logging

from typing import Tuple


logger = logging.getLogger(__name__)
rng = np.random.default_rng()

# Use of simulated annealing to produce a near optimal solution for the travelling sales person problem

def solve_tsp_brute_force(cities: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Brute forces through all permutations to find shortest path
    :param cities: list of points in 2D space that we want shortest path through
    :returns: path and length of path
    """
    # go through all permutations and get one with lowest distance
    permutations = itertools.permutations(cities)
    all_paths = [[city for city in permutation] for permutation in permutations]
    all_dists = []
    for path in all_paths:
        path_dist = total_dist(np.array(path))
        all_dists.append(path_dist)
    idx = np.argmax(all_dists)
    return np.array(all_paths[idx]), all_dists[idx]

# to improve this we could branch out and work on various paths per iteration
def solve_tsp_simulated_annealing(cities: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Returns near optimal solution for travelling salesperson problem via simulated annealing
    :param cities: list of points in 2D space that we want shortest path through
    :returns: path and total length of path
    """
    N = 10000
    cur_path = rng.permutation(cities)
    path_len = len(cur_path)
    f = lambda x: 0.1*np.exp(x*(-9.21/(N-1))) # schedule starting at 1e-1 and ending at 1e-5 following exponential decay
    temp_schedule = list(map(f, range(N)))
    for t in temp_schedule:
        i,j = rng.choice(range(path_len), size=2, replace=False)
        new_path = cur_path.copy()
        new_path[i], new_path[j] = new_path[j], new_path[i]
        cur_dist, new_dist = total_dist(cur_path), total_dist(new_path)

        new_path_prob = np.exp((cur_dist - new_dist)/t)
        # using probability of path ~= e^(-C(tour)/t) -- the lower the cost the higher the probability
        if 1 <= new_path_prob:
            cur_path = new_path
        else:
            q = 1 - new_path_prob
            chosen_path = rng.choice(a=[cur_path, new_path], size=1, p=[new_path_prob, q])[0]
            cur_path = chosen_path
    return cur_path, total_dist(cur_path)

def total_dist(path: np.ndarray) -> float:
    cycle = np.append(path, path[0])
    total_dist = 0
    for a,b in itertools.pairwise(cycle):
        dist = np.linalg.norm(a-b)
        total_dist += dist
    return total_dist

def main(cities_num: int):
    N = 20
    distance_differences = []
    for i in range(N):
        cities = rng.uniform(size=(cities_num, 2))
        _, bf_dist = solve_tsp_brute_force(cities)
        _, tsp_dist = solve_tsp_simulated_annealing(cities)
        # logger.info(f"brute force path distance {bf_dist} vs tsp path distance {tsp_dist}")
        print(f"brute force path distance {bf_dist} vs tsp path distance {tsp_dist}")
        distance_differences.append(abs(bf_dist - tsp_dist))
    avg_difference = sum(distance_differences)/len(distance_differences)
    # logger.info(f"avg difference between brute force and tsp path was {avg_difference}")
    print(f"avg difference between brute force and tsp path was {avg_difference}")

if __name__ == '__main__':
    main(cities_num=5)
