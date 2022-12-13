import numpy as np
import ecdf_estimator.utils as ecdf_aux


def estimate_radii_values( dataset_a, dataset_b, distance_fct, eps = 0.05, rel_offset = 0.05 ):
  distance_data = [ distance_fct(data_a, data_b) for data_b in dataset_b for data_a in dataset_a ]
  while isinstance(distance_data[0], list):
    distance_data = [item for sublist in distance_data for item in sublist]
  distance_data = np.sort(distance_data)

  data_offset = round(len(distance_data) * rel_offset)
  radius_max = distance_data[-(data_offset + 1)]
  radius_min = distance_data[data_offset]

  upper_bound = radius_max + eps * (radius_max - radius_min)
  lower_bound = radius_min - eps * (radius_max - radius_min)
  if lower_bound < 0:
    lower_bound = radius_min

  return lower_bound, upper_bound, distance_data


def choose_bins(distance_data, possible_bins, n_bins = 10, min_value_shift = "default",
  max_value_shift = "default", choose_type = "uniform_y" ):
  ecdf_curve = ecdf_aux.empirical_cumulative_distribution_vector(distance_data, possible_bins)
  if choose_type == "uniform_y":
    max_value, min_value = np.amax( ecdf_curve ), np.amin( ecdf_curve )
    if min_value_shift == "default":  min_value_shift = (max_value - min_value) / n_bins
    if max_value_shift == "default":  max_value_shift = (min_value - max_value) / n_bins
    rad_bdr   = np.linspace( min_value+min_value_shift , max_value+max_value_shift , num=n_bins )
    indices   = [ np.argmax( ecdf_curve >= bdr ) for bdr in rad_bdr ]
    unique_indices = np.unique(indices)
    if len(indices) != len(unique_indices):
      print("WARNING: Some bins were duplicate. These duplicates are removed from the list.")
    return [ possible_bins[i] for i in unique_indices ]
  elif choose_type == "uniform_x":
    max_index, min_index = np.amax( np.argmin(ecdf_curve) ), np.amin( np.argmax(ecdf_curve) )
    if min_value_shift == "default":  min_value_shift = (max_index - min_index) / n_bins
    if max_value_shift == "default":  max_value_shift = (min_index - max_index) / n_bins
    indices   = np.linspace( min_index+min_value_shift , max_index+max_value_shift , num=n_bins )
    unique_indices = np.unique(indices)
    if len(indices) != len(unique_indices):
      print("WARNING: Some bins were duplicate. These duplicates are removed from the list.")
    return [ possible_bins[int(i)] for i in unique_indices ]
  else:
    print("WARNING: Invalid choose_type flag for choose_bins. Nothing is done in this function.")
