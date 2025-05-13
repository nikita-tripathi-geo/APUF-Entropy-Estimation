#!/usr/bin/python3
"""This module has various helper functions for APUF entropy estimation.
"""

import numpy as np
import apuf_simulation as puf
from itertools import combinations
import sys, pickle
from multiprocessing import Pool
import matplotlib.pyplot as plt


def read_response_sequences(
    apuf_list: np.ndarray,
    chal_seq: np.ndarray,
    measurement_num:int = 100) -> list[np.ndarray]:
  """ Reads sequences of responses from a list of APUFs.
  -
  Works for both - uniqueness and extractable entropy.\n
  Takes a list of APUFs with same number of layers.\n

  """
  # For all APUFs store a list of their responses as an ndarray
  resp_sequences = []
  # read from APUF i
  for apuf in apuf_list:
    resp = []     # response of APUF i
    for chal in chal_seq:
      # read measurement_num times
      resp.extend(
        [puf.get_noisy_responses(1, [apuf], chal, noise_mean=0, noise_std=0.005)
          for _ in range(measurement_num)]
      )
    resp_sequences.append(np.concatenate(resp))

  return resp_sequences


def read_responses_ext(
    apuf: np.ndarray,
    chall_seq: list[np.ndarray],
    m: int = 100
) -> list[np.ndarray]:
  """Reads from `n` segments of APUF of length `k`.

  Here `n = len(chal_seq)` and `k = len(chal_seq[0])`
  `m` is the number of measurements for each segment

  Outputs a list of n segments, each of which is read m times
  ...
  """
  # For each segment of the APUF, store m readings as an ndarray
  segments = []

  # We only have 1 APUF, but many possible challenges
  for chal in chall_seq:
    # chal is a block of $k$ challenges
    resp = [puf.get_noisy_responses(1, [apuf], chal, noise_mean=0, noise_std=0.005)
          for _ in range(m)]

    segments.append(np.concatenate(resp))

  return segments



def generate_distances(resp_sequences: list[np.ndarray]) -> list[np.floating]:
  """Generates distances between responses.

  Finds distances between every pair of APUF responses.
  Returns list of distances, stored as numpy floats.

  Args:
    resp_sequences:
      A list of NumPy ndarray-s. Each ndarray contains a sequence of
      authentication responses (A_resp) from one APUF.

  Returns:
    Distances - a list of NumPy floats. Each represents a fractional
    Hamming Distance between A_resps from different ndarrays (different
    elements of resp_sequences).

  Raises:
    Nothing: N/A.
  """

  results = []

  # find response length
  resp_len = len(resp_sequences[0][0])

  # start pairwise comparisons.
  # 1) first != second
  # 2) (first_i, second_i != second_j, first_j) for any i, j
  for first, second in combinations(resp_sequences, 2):
    # XOR pair - produces a ndarray of bools (same dimensions)
    comp = first == second

    # Find fractional Hamming distances
    distances = [ (resp_len - np.sum(response)) / resp_len
        for response in comp]

    results.extend(distances)

  return results


def shannon_entropy(p: np.floating) -> np.floating:
  """Calculates shannon entropy in the binary case.

  Args:
      p (np.floating): Probability of a random event occuring. Since we consider
      APUF responses, we assume there are only two possible events (response bit
      is 0 or 1).

  Returns:
      np.floating: Shannon entropy of random event.
  """
  p_inverse = 1 - p
  return -1 * (p * np.log2(p) + p_inverse * np.log2(p_inverse))


def degrees_and_entropy(
    distances: list[np.floating]
    ) -> tuple[np.floating, np.floating]:
  """Calculates degrees-of-freedom and estimated entropy.

  Uses Daugman's method (TODO WRITE MORE)
  """

  mean = np.mean(distances)
  variance = np.var(distances)

  degrees_of_freedom = (1 - mean) * mean / variance

  entropy = shannon_entropy(mean) * degrees_of_freedom

  return degrees_of_freedom, entropy


def unit_challenge_generation(params: tuple[int, int, int]) -> np.ndarray:
  """Single process for parallel computation of many challenges.

  Args:
      response_len (int): Length of the desired response.

  Returns:
      np.ndarray: Sequence of `response_len` challenges.
  """
  response_len, challenge_len, seed = params
  return puf.generate_n_challenges(response_len, challenge_len, seed)


def main():
  """For testing purposes only.

  Run as:
  ./entropy_helper.py layers resp-len num-of-resps measure-num num-of-apufs
  """
  if sys.argv[1:] == []:
    layers = 128 + 1
    resplen = 100
    respnum = 1000
    measurement_num = 100
    apufnum = 100
  else:
    # no error checking...
    print(sys.argv)
    assert len(sys.argv[1:]) == 5

    layers, resplen, respnum, measurement_num, apufnum = (int(arg) for arg in sys.argv[1:])

    # print(layers, resplen, respnum, measurement_num, apufnum)

  print(f"Layers: {layers - 1}\nResponse length: {resplen}")


  with Pool() as p:
    # chals = p.map(unit_challenge_generation, [(resplen, layers)]*respnum)
    chals = p.map(unit_challenge_generation, [(resplen, layers, seed) for seed in range(respnum)])

  # chals = [puf.generate_n_challenges(resplen, layers) for _ in range(respnum)]

  print(len(chals), chals[0].shape)

  DFs, Hs = [], []    # pylint: disable=invalid-name

  for _ in range(10):
    apufs_uniq = [
      puf.generate_n_APUFs(1, layers, weight_mean=0, weight_stdev=0.05)
      for _ in range(apufnum)
      ]
    # apuf = puf.generate_n_APUFs(1,layers, 0, 0.05)
    # apufs_ext = [apuf for _ in range(apufnum)]

    # d_ext, h_ext = degrees_and_entropy(
    #   generate_distances(read_response_sequences(apufs_ext, chals, measurement_num))
    #   )
    # TODO: Do this for N APUFs
    # results = [degrees_and_entropy(
    #   generate_distances(read_responses_ext(apuf, chals, measurement_num))
    # ) for apuf in apufs_uniq]

    distances = [
      generate_distances(read_responses_ext(apuf, chals, measurement_num)) for apuf in apufs_uniq
    ]

    flat_distances = [d for ds in distances for d in ds]
    # print("Number of data points: ", len(flat_distances))
    d_ext, h_ext = degrees_and_entropy(flat_distances)
    
    print([degrees_and_entropy(d) for d in distances])

    # plot = plt.hist(flat_distances, bins='auto')
    # plt.savefig("graph2.png")
    # plt.show()
    # TODO take this outside of the loop?

    # for d_ext, h_ext in results:
    #   Hs.append(h_ext)
    #   DFs.append(d_ext)

    # d_uniq, h_uniq = degrees_and_entropy(
    #   generate_distances(read_response_sequences(apufs_uniq, chals, measurement_num))
    #   )

    # DFs.append((d_ext, d_uniq))
    # Hs.append((h_ext, h_uniq))
    # Hs.append(h_uniq)
    # DFs.append(d_uniq)
    Hs.append(h_ext)
    DFs.append(d_ext)

  # print(f"Extractable DF = {np.mean([i[0] for i in DFs])}\nUniqueness DF = {np.mean([i[1] for i in DFs])}")
  # print(f"Extractable Entropy = {np.mean([i[0] for i in Hs])}\nUniqueness Entropy = {np.mean([i[1] for i in Hs])}")
  print(Hs)
  print(f"Extractable entropy for k={resplen}: {np.mean(Hs)}")
  print(DFs)
  print(f"Degrees of freedom: {np.mean(DFs)}")


if __name__ == "__main__":
  main()
