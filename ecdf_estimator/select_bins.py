import numpy as np
import ecdf_estimator.utils as ecdf_aux


## \brief  Heuristically determine region in which useful bin/radii values are located.
#
#  \param   dataset_a        Subset of the overall dataset.
#  \param   dataset_b        Different subset of the poverall dataset.
#  \param   distance_fct     Functions that evaluates (generalized) distance between subset members.
#  \param   rel_offset       Relative offset to determin interval of reasonable bin values.
#  \param   rel_cutoff       Relative cutoff of the interval of reasonable bin values.            
#  \retval  target_val       The value of the target function.
def estimate_radii_values( dataset_a, dataset_b, distance_fct, rel_offset=0.05, rel_cutoff=0.05 ):
  distance_data = [ distance_fct(data_a, data_b) for data_b in dataset_b for data_a in dataset_a ]
  while isinstance(distance_data[0], list):
    distance_data = [item for sublist in distance_data for item in sublist]
  distance_data = np.sort(distance_data)

  data_offset = round(len(distance_data) * rel_offset)
  radius_max  = distance_data[-(data_offset + 1)]
  radius_min  = distance_data[data_offset]

  upper_bound = radius_max + rel_cutoff * (radius_max - radius_min)
  lower_bound = radius_min - rel_cutoff * (radius_max - radius_min)
  if lower_bound < 0:
    lower_bound = radius_min

  return lower_bound, upper_bound, distance_data


## \brief  Heuristically determine reasonable bin/radii values from larger choice.
#
#  \param   distance_list    List of distances to be grouped into the bins.
#  \param   bins             List of possible bin values.
#  \param   n_bins           Maximum amount of bins, which are to be selected. Defaults to 10.
#  \param   choose_typ       Heurustic that is used to select bins.
#  \param   min_value_shift  Exclude values that are smaller than min value of distances plus this.
#  \param   max_value_shift  Exclude values that are largr than max value of distances plus this.
#  \retval  target_val       The value of the target function.
def choose_bins(distance_list, possible_bins, n_bins=10, choose_type="uniform_y",
  min_value_shift=None, max_value_shift=None ):
  ecdf_curve = ecdf_aux.empirical_cumulative_distribution_vector(distance_list, possible_bins)
  if choose_type == "uniform_y":
    max_value, min_value = np.amax( ecdf_curve ), np.amin( ecdf_curve )
    if min_value_shift is None:  min_value_shift = (max_value - min_value) / n_bins
    if max_value_shift is None:  max_value_shift = (min_value - max_value) / n_bins
    rad_bdr   = np.linspace( min_value+min_value_shift , max_value+max_value_shift , num=n_bins )
    indices   = [ np.argmax( ecdf_curve >= bdr ) for bdr in rad_bdr ]
    unique_indices = np.unique(indices)
    if len(indices) != len(unique_indices):
      print("WARNING: Some bins were duplicate. These duplicates are removed from the list.")
    return [ possible_bins[i] for i in unique_indices ]
  elif choose_type == "uniform_x":
    max_index, min_index = np.amax( np.argmin(ecdf_curve) ), np.amin( np.argmax(ecdf_curve) )
    if min_value_shift is None:  min_value_shift = (max_index - min_index) / n_bins
    if max_value_shift is None:  max_value_shift = (min_index - max_index) / n_bins
    indices   = np.linspace( min_index+min_value_shift , max_index+max_value_shift , num=n_bins )
    unique_indices = np.unique(indices)
    if len(indices) != len(unique_indices):
      print("WARNING: Some bins were duplicate. These duplicates are removed from the list.")
    return [ possible_bins[int(i)] for i in unique_indices ]
  else:
    print("WARNING: Invalid choose_type flag for choose_bins. Nothing is done in this function.")
