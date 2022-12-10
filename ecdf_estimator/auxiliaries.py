import numpy as np


def empirical_cumulative_distribution_vector( distance_list, bins ):
  # n_close_elem = [0.] * len(bins)                   # Use of np.zeros( len(bins) ) slows down code!
  # for distance in distance_list:                                                                  # This code is perfectly fine and the intuitve
  #   n_close_elem = [ ( n_close_elem[i] + (distance < bins[i]) ) for i in range(len(bins)) ]       # solution. However, it is slow.
  return [ np.sum( [distance < basket for distance in distance_list] ) / len(distance_list) \
    for basket in bins ]


def create_distance_matrix( dataset_a, dataset_b, distance_fct, 
  start_a = 0, end_a = -1, start_b = 0, end_b = -1 ):
  if end_a == -1:  end_a = len(dataset_a)
  if end_b == -1:  end_b = len(dataset_b)

  # init_val        = distance_fct(dataset_a[start_a], dataset_b[start_b])                          # Perfectly fine code that does the job, but uses
  # distance_tensor = [[init_val for _ in range(end_b - start_b) ] for _ in range(end_a - start_a)] # 1 more eval than necessary.
  # for i in range(end_a - start_a):
  #   for j in range(end_b - start_b):
  #     distance_tensor[i][j] = distance_fct(dataset_a[start_a+i], dataset_b[start_b+j])

  return [ [ distance_fct(dataset_a[start_a+i], dataset_b[start_b+j]) \
    for j in range(end_b - start_b) ] for i in range(end_a - start_a) ]


def empirical_cumulative_distribution_vector_list( dataset, bins, distance_fct, subset_indices ):
  if not all(subset_indices[i] <= subset_indices[i+1] for i in range(len(subset_indices)-1)):
    raise Exception("Subset indices are out of order.")
  if subset_indices[0] != 0 or subset_indices[-1] != len(dataset):
    raise Exception("Not all elements of the dataset are distributed into subsets.")

  matrix = []
  for i in range(len(subset_indices)-1):
    for j in range(i):
      distance_list = create_distance_matrix(dataset, dataset, distance_fct, 
        subset_indices[i], subset_indices[i+1], subset_indices[j], subset_indices[j+1])
      while isinstance(distance_list[0], list):
        distance_list = [item for sublist in distance_list for item in sublist]
      matrix.append( empirical_cumulative_distribution_vector(distance_list, bins) )

  return np.transpose(matrix)


def empirical_cumulative_distribution_vector_list_bootstrap(
  dataset_a, dataset_b, bins, distance_fct, n_samples):
  distance_matrix = np.array( create_distance_matrix(dataset_a, dataset_b, distance_fct) )
  matrix = []
  for _ in range(n_samples):
    permute_a = np.random.randint(distance_matrix.shape[0], size=distance_matrix.shape[0])
    permute_b = np.random.randint(distance_matrix.shape[1], size=distance_matrix.shape[1])
    distance_list = np.ndarray.flatten( distance_matrix[permute_a,permute_b] )
    matrix.append( empirical_cumulative_distribution_vector(distance_list, bins) )
  return np.transpose(matrix)


def mean_of_ecdf_vectors( ecdf_vector_list ):
  return [ np.mean(vector) for vector in ecdf_vector_list ]


def covariance_of_ecdf_vectors( ecdf_vector_list ):
  return np.cov( ecdf_vector_list )


def file_output( obj_fun ):
  np.savetxt(obj_fun.type + '_bins.txt',         obj_fun.bins, fmt='%.6f')
  np.savetxt(obj_fun.type + '_ecdf-list.txt',    obj_fun.ecdf_list, fmt='%.6f')
  np.savetxt(obj_fun.type + '_mean-vector.txt',  obj_fun.mean_vector, fmt='%.6f')
  np.savetxt(obj_fun.type + '_covar-matrix.txt', obj_fun.covar_matrix, fmt='%.6f')


def evaluate_from_empirical_cumulative_distribution_functions( obj_fun, vector ):
  mean_deviation = np.subtract( obj_fun.mean_vector , vector )
  try:
    return np.dot( mean_deviation , np.linalg.solve(obj_fun.covar_matrix, mean_deviation) )
  except np.linalg.LinAlgError as error:
    if not obj_fun.error_printed:
      obj_fun.error_printed = True
      print("WARNING: Covariance matrix is singular. CIL_estimator uses different topology.")
    return np.dot( mean_deviation , mean_deviation )


def choose_bins(obj_fun, n_bins = 10, min_value_shift = "default", max_value_shift = "default",
  choose_type = "uniform_y", check_spectral_conditon = True, file_output = False ):
  if choose_type == "uniform_y":
    max_value = np.amax( obj_fun.mean_vector )
    min_value = np.amin( obj_fun.mean_vector )
    if min_value_shift == "default":  min_value_shift = (max_value - min_value) / n_bins
    if max_value_shift == "default":  max_value_shift = (min_value - max_value) / n_bins
    rad_bdr   = np.linspace( min_value+min_value_shift , max_value+max_value_shift , num=n_bins )
    indices   = [ np.argmax( obj_fun.mean_vector >= bdr ) for bdr in rad_bdr ]
    unique_indices = np.unique(indices)
    if len(indices) != len(unique_indices):
      print("WARNING: Some bins were duplicate. These duplicates are removed from the list.")
    obj_fun.bins = [ obj_fun.bins[i] for i in unique_indices ]
  elif choose_type == "uniform_x":
    max_index = np.amax( np.argmin(obj_fun.mean_vector) )
    min_index = np.amin( np.argmax(obj_fun.mean_vector) )
    if min_value_shift == "default":  min_value_shift = (max_index - min_index) / n_bins
    if max_value_shift == "default":  max_value_shift = (min_index - max_index) / n_bins
    indices   = np.linspace( min_index+min_value_shift , max_index+max_value_shift , num=n_bins )
    unique_indices = np.unique(indices)
    if len(indices) != len(unique_indices):
      print("WARNING: Some bins were duplicate. These duplicates are removed from the list.")
    obj_fun.bins = [ obj_fun.bins[int(i)] for i in unique_indices ]
  else:
    print("WARNING: Invalid choose_type flag for choose_bins. Nothing is done in this function.")
    return

  if obj_fun.type == "standard" or obj_fun.type == "distribution":
    obj_fun.ecdf_list = empirical_cumulative_distribution_vector_list(
      obj_fun.dataset, obj_fun.bins, obj_fun.distance_fct, obj_fun.subset_indices )
  elif obj_fun.type == "bootstrap":
    obj_fun.ecdf_list = empirical_cumulative_distribution_vector_list_bootstrap(
      obj_fun.dataset_a, obj_fun.dataset_b, obj_fun.bins, obj_fun.distance_fct, obj_fun.n_samples )
  else:
    print("ERROR: Internal error detected.")
    return

  obj_fun.mean_vector   = mean_of_ecdf_vectors(obj_fun.ecdf_list)
  obj_fun.covar_matrix  = covariance_of_ecdf_vectors(obj_fun.ecdf_list)
  if file_output:
    np.savetxt('choose-bins_bins.txt', obj_fun.bins, fmt='%.6f')
    np.savetxt('choose-bins_ecdf-list.txt', obj_fun.ecdf_list, fmt='%.6f')
    np.savetxt('choose-bins_mean-vector.txt', obj_fun.mean_vector, fmt='%.6f')
    np.savetxt('choose-bins_covar-matrix.txt', obj_fun.covar_matrix, fmt='%.6f')
  if check_spectral_conditon:
    spectral_condition = np.linalg.cond(obj_fun.covar_matrix)
    if spectral_condition > 1e3:
      print("WARNING: The spectral condition of the covariance matrix is", spectral_condition)
