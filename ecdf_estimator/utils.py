import numpy as np


# Use this function to estimate good bin values BEFORE creating objective function.


# estimates minimal and maximal radii values based on a small segment of pattern data
def estimate_radii_values(
    dataset1,  # first subset of the data (first two parameters can be tuned to mimic the behaviour of full dataset)
    dataset2,  # second subset of the data
    distance_function,  # function, used to compute distances between patterns
    eps=0.05,  # small constant, used to perturb the estimates
    rel_offset=0.05  # percentage of distances from the 'tails', which will be skipped (from one side)
  ):
  n1 = len(dataset1)
  n2 = len(dataset2)
  distance_data = np.ndarray((n1, n2))
  for i in range(n1):
    for j in range(n2):
      distance_data[i, j] = distance_function(dataset1[i], dataset2[j])
  distance_data = distance_data.flatten()
  distance_data = np.sort(distance_data)
  data_offset = round(len(distance_data) * rel_offset)
  r_max = distance_data[-(data_offset + 1)]
  r_min = distance_data[data_offset]

  radii_interval = (r_max - r_min)
  upper_bound = r_max + eps * radii_interval
  lower_bound = r_min - eps * radii_interval
  if lower_bound < 0:
    lower_bound = r_min
  return lower_bound, upper_bound, radii_interval
