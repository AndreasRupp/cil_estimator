import numpy as np


def estimate_radii_values( dataset1, dataset_b, distance_function, eps = 0.05, rel_offset = 0.05 ):
  distance_data = [ distance_function(dataset_a[i], dataset_b[j]) for j in range(len(dataset_b)) \
                    for i in range(len(dataset_a)) ]
  if isinstance(distance_data[0], list):
    distance_data = [item for sublist in distance_data for item in sublist]
  distance_data = np.sort(distance_data)

  data_offset = round(len(distance_data) * rel_offset)
  r_max = distance_data[-(data_offset + 1)]
  r_min = distance_data[data_offset]

  radii_interval = (r_max - r_min)
  upper_bound = r_max + eps * radii_interval
  lower_bound = r_min - eps * radii_interval
  if lower_bound < 0:
    lower_bound = r_min

  return lower_bound, upper_bound, distance_data


def choose_bins(distance_data, possible_bins, n_bins = 10, min_value_shift = "default",
  max_value_shift = "default", choose_type = "uniform_y", check_spectral_conditon = True,
  file_output = False ):
  ecdf_curve = empirical_cumulative_distribution_vector(distance_data, possible_bins)
  if choose_type == "uniform_y":
    max_value = np.amax( ecdf_curve )
    min_value = np.amin( ecdf_curve )
    if min_value_shift == "default":  min_value_shift = (max_value - min_value) / n_bins
    if max_value_shift == "default":  max_value_shift = (min_value - max_value) / n_bins
    rad_bdr   = np.linspace( min_value+min_value_shift , max_value+max_value_shift , num=n_bins )
    indices   = [ np.argmax( ecdf_curve >= bdr ) for bdr in rad_bdr ]
    unique_indices = np.unique(indices)
    if len(indices) != len(unique_indices):
      print("WARNING: Some bins were duplicate. These duplicates are removed from the list.")
    return [ possible_bins[i] for i in unique_indices ]
  elif choose_type == "uniform_x":
    max_index = np.amax( np.argmin(ecdf_curve) )
    min_index = np.amin( np.argmax(ecdf_curve) )
    if min_value_shift == "default":  min_value_shift = (max_index - min_index) / n_bins
    if max_value_shift == "default":  max_value_shift = (min_index - max_index) / n_bins
    indices   = np.linspace( min_index+min_value_shift , max_index+max_value_shift , num=n_bins )
    unique_indices = np.unique(indices)
    if len(indices) != len(unique_indices):
      print("WARNING: Some bins were duplicate. These duplicates are removed from the list.")
    return [ possible_bins[int(i)] for i in unique_indices ]
  else:
    print("WARNING: Invalid choose_type flag for choose_bins. Nothing is done in this function.")
    return
