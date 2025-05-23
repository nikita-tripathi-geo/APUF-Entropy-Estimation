"""TODO"""

# Imports
from itertools import combinations

import numpy as np
from apuf import APUF
from challenges import generate_challenges_mp


def U1(     # pylint: disable=invalid-name
    apuf_pair: tuple[APUF, APUF],
    challenges: np.ndarray,
    m: int = 50,
    noise: tuple[float, float] = (0.0, 0.005),
) -> np.ndarray:
    """TODO"""
    apuf_a, apuf_b = apuf_pair
    dist = np.empty(m, dtype=np.float64)

    for i in range(m):
        resp_a = apuf_a.get_responses(challenges, *noise)
        resp_b = apuf_b.get_responses(challenges, *noise)

        dist[i] = resp_a.fhd(resp_b)

    return dist


def U2(     # pylint: disable=invalid-name
    apuf_pair: tuple[APUF, APUF],
    chal_params: tuple[int, int],
    m: int,
    noise: tuple[float, float] = (0.0, 0.005),
    seed: int = 0
) -> np.ndarray:
    """TODO"""
    k, Ns = chal_params
    # Preallocate memory
    dist = np.empty(Ns * m, dtype=np.float64)
    # Ns * k random challenges using a multiprocessing library
    challenges = generate_challenges_mp(Ns, k, apuf_pair[0].d - 1, seed)

    for i, chal in enumerate(challenges):
        dist[i : i + m] = U1(apuf_pair, chal, m, noise)

    return dist


# U3
def U3(     # pylint: disable=invalid-name
    puf_params: tuple[int, int],
    chal_params: tuple[int, int],
    m: int,
    weights: tuple[float, float] = (0.0, 0.05),
    noise: tuple[float, float] = (0.0, 0.005),
    seed: int = 0,
) -> np.ndarray:
    """TODO"""

    # unpack APUF parameters
    n, d = puf_params

    # simulate n apufs
    pufs = [APUF(d, *weights) for _ in range(n)]

    # setup
    dist = []

    for pair in combinations(pufs, 2):
        dist.extend(U2(pair, chal_params, m, noise, seed))

    return np.array(dist)


def main():
    args = {
        "puf_params" : (30, 128),
        "chal_params" : (200, 100),
        "m" : 100,
        "weights" : (0.0, 0.05),
        "noise" : (0.0, 0.005),
        "seed" : 0
    }

    results = U3(**args)

    print(np.mean(results), np.std(results))
    print(results.shape, results)


if __name__ == "__main__":
    main()
