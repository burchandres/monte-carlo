import numpy as np
import unittest

from tsp import solve_tsp_simulated_annealing, solve_tsp_brute_force

class TSP(unittest.TestCase):

    def testTSP(self):
        rng = np.random.default_rng()
        N = 10 # number of times to run tsp solve functions
        num_of_cities = 10 # 10! permutations to go through so it'll take a while

        for _ in range(N):
            cities = rng.uniform(size=(num_of_cities, 2))
            _, bf_dist = solve_tsp_brute_force(cities)
            _, tsp_dist = solve_tsp_simulated_annealing(cities)
            # brute force should always produce optimum solution
            self.assertLessEqual(bf_dist, tsp_dist)

if __name__ == '__main__':
    unittest.main()