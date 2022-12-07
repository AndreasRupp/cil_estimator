import numpy as np


def empirical_cumulative_distribution_vector( distance_list, bins ):
  n_close_elem = [0.] * len(bins)            # Use of np.zeros( len(bins) ) slows down code!
  for distance in distance_list:
    n_close_elem = [ ( n_close_elem[i] + (distance < bins[i]) ) for i in range(len(bins)) ]
  return [ elem / len(distance_list) for elem in n_close_elem ]


def create_distance_matrix( dataset_a, dataset_b, distance_fct, 
  start_a = 0, end_a = -1, start_b = 0, end_b = -1 ):
  if end_a == -1:  end_a = len(dataset_a)
  if end_b == -1:  end_b = len(dataset_b)

  distance_list = np.zeros( (end_a-start_a ,end_b-start_b) )

  for i in range(end_a - start_a):
    for j in range(end_b - start_b):
      distance_list[i,j] = distance_fct(dataset_a[start_a+i], dataset_b[start_b+j])

  return distance_list


def empirical_cumulative_distribution_vector_list( dataset, bins, distance_fct, subset_indices ):
  if not all(subset_indices[i] <= subset_indices[i+1] for i in range(len(subset_indices)-1)):
    raise Exception("Subset indices are out of order.")
  if subset_indices[0] != 0 or subset_indices[-1] != len(dataset):
    raise Exception("Not all elements of the dataset are distributed into subsets.")

  matrix = []
  for i in range(len(subset_indices)-1):
    for j in range(i):
      distance_list = np.ndarray.flatten( create_distance_matrix(dataset, dataset, distance_fct, 
        subset_indices[i], subset_indices[i+1], subset_indices[j], subset_indices[j+1]) )
      matrix.append( empirical_cumulative_distribution_vector(distance_list, bins) )
  return np.transpose(matrix)


def empirical_cumulative_distribution_vector_list_bootstrap(
  dataset_a, dataset_b, bins, distance_fct, n_samples):
  distance_matrix = create_distance_matrix(dataset_a, dataset_b, distance_fct)
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




def _choose_bins(obj_fun, n_bins = 10, min_value_shift = "default", max_value_shift = "default",
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

  if obj_fun.type == "standard":
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


def _evaluate_from_empirical_cumulative_distribution_functions( obj_fun, vector ):
  mean_deviation = np.subtract( obj_fun.mean_vector , vector )
  try:
    return np.dot( mean_deviation , np.linalg.solve(obj_fun.covar_matrix, mean_deviation) )
  except np.linalg.LinAlgError as error:
    if not obj_fun.error_printed:
      obj_fun.error_printed = True
      print("WARNING: Covariance matrix is singular. CIL_estimator uses different topology.")
    return np.dot( mean_deviation , mean_deviation )





class objective_function:
  def __init__( self, dataset, bins, distance_fct, subset_sizes, file_output = False ):
    self.type           = "standard"
    self.dataset        = dataset
    self.bins           = bins
    self.distance_fct   = distance_fct
    self.subset_indices = [ sum(subset_sizes[:i]) for i in range(len(subset_sizes)+1) ]
    self.ecdf_list      = empirical_cumulative_distribution_vector_list(
                            dataset, bins, distance_fct, self.subset_indices )
    self.mean_vector    = mean_of_ecdf_vectors(self.ecdf_list)
    self.covar_matrix   = covariance_of_ecdf_vectors(self.ecdf_list)
    self.error_printed  = False
    if file_output:
      np.savetxt('obj-func_bins.txt', self.bins, fmt='%.6f')
      np.savetxt('obj-func_ecdf-list.txt', self.ecdf_list, fmt='%.6f')
      np.savetxt('obj-func_mean-vector.txt', self.mean_vector, fmt='%.6f')
      np.savetxt('obj-func_covar-matrix.txt', self.covar_matrix, fmt='%.6f')

  def choose_bins( self, n_bins = 10, min_value_shift = "default", max_value_shift = "default",
    choose_type = "uniform_y", check_spectral_conditon = True, file_output = False ):
    _choose_bins(self, n_bins, min_value_shift, max_value_shift, choose_type,
      check_spectral_conditon, file_output)

  def evaluate_from_empirical_cumulative_distribution_functions( self, vector ):
    return _evaluate_from_empirical_cumulative_distribution_functions( self, vector )

  def evaluate( self, dataset ):
    comparison_set = np.random.randint( len(self.subset_indices)-1 )
    distance_list = np.ndarray.flatten( create_distance_matrix(self.dataset, dataset, 
      self.distance_fct, self.subset_indices[comparison_set],
      self.subset_indices[comparison_set+1]) )
    y = empirical_cumulative_distribution_vector(distance_list,self.bins)
    return self.evaluate_from_empirical_cumulative_distribution_functions( y )


class bootstrap_objective_function:
  def __init__( self, dataset_a, dataset_b, bins, distance_fct, n_samples = 1000, 
    file_output = False ):
    self.type           = "bootstrap"
    self.dataset_a      = dataset_a
    self.dataset_b      = dataset_b
    self.bins           = bins
    self.distance_fct   = distance_fct
    self.n_samples      = n_samples
    self.ecdf_list      = empirical_cumulative_distribution_vector_list_bootstrap(
                            dataset_a, dataset_b, bins, distance_fct, self.n_samples )
    self.mean_vector    = mean_of_ecdf_vectors(self.ecdf_list)
    self.covar_matrix   = covariance_of_ecdf_vectors(self.ecdf_list)
    self.error_printed  = False
    if file_output:
      np.savetxt('obj-func_bins.txt', self.bins, fmt='%.6f')
      np.savetxt('obj-func_ecdf-list.txt', self.ecdf_list, fmt='%.6f')
      np.savetxt('obj-func_mean-vector.txt', self.mean_vector, fmt='%.6f')
      np.savetxt('obj-func_covar-matrix.txt', self.covar_matrix, fmt='%.6f')

  def choose_bins( self, n_bins = 10, min_value_shift = "default", max_value_shift = "default",
    choose_type = "uniform_y", check_spectral_conditon = True, file_output = False ):
    _choose_bins(self, n_bins, min_value_shift, max_value_shift, choose_type,
      check_spectral_conditon, file_output)

  def evaluate_from_empirical_cumulative_distribution_functions( self, vector ):
    return _evaluate_from_empirical_cumulative_distribution_functions( self, vector )

  def evaluate( self, dataset ):
    if np.random.randint( 2 ) == 0:
      comparison_set = self.dataset_a
    else:
      comparison_set = self.dataset_b

    distance_list = np.ndarray.flatten( create_distance_matrix(comparison_set, dataset, 
      self.distance_fct) )
    y = empirical_cumulative_distribution_vector(distance_list,self.bins)
    return self.evaluate_from_empirical_cumulative_distribution_functions( y )
