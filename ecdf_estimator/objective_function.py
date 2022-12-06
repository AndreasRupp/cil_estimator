import numpy as np


def empirical_cumulative_distribution_vector( dataset_a, dataset_b, bins, distance_fct, 
  start_a = 0, end_a = -1, start_b = 0, end_b = -1 ):
  if end_a == -1:  end_a = len(dataset_a)
  if end_b == -1:  end_b = len(dataset_b)

  n_close_elem = [0.] * len(bins)            # Use of np.zeros( len(bins) ) slows down code!
  for elem_a in dataset_a[start_a:end_a]:
    for elem_b in dataset_b[start_b:end_b]:
      distance = distance_fct(elem_a, elem_b)
      n_close_elem = [ ( n_close_elem[i] + (distance < bins[i]) ) for i in range(len(bins)) ]
  return [ elem / ((end_a - start_a) * (end_b - start_b)) for elem in n_close_elem ]


def empirical_cumulative_distribution_vector_list( dataset, bins, distance_fct, subset_indices ):
  if not all(subset_indices[i] <= subset_indices[i+1] for i in range(len(subset_indices)-1)):
    raise Exception("Subset indices are out of order.")
  if subset_indices[0] != 0 or subset_indices[-1] != len(dataset):
    raise Exception("Not all elements of the dataset are distributed into subsets.")

  matrix = []
  for i in range(len(subset_indices)-1):
    for j in range(i):
      matrix.append( empirical_cumulative_distribution_vector(dataset, dataset, bins, distance_fct,
        subset_indices[i], subset_indices[i+1], subset_indices[j], subset_indices[j+1]) )
  return np.transpose(matrix)


def mean_of_ecdf_vectors( ecdf_vector_list ):
  return [ np.mean(vector) for vector in ecdf_vector_list ]


def covariance_of_ecdf_vectors( ecdf_vector_list ):
  return np.cov( ecdf_vector_list )


class objective_function:
  def __init__( self, dataset, bins, distance_fct, subset_sizes, file_output = False ):
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
    if choose_type == "uniform_y":
      max_value = np.amax( self.mean_vector )
      min_value = np.amin( self.mean_vector )
      if min_value_shift == "default":  min_value_shift = (max_value - min_value) / n_bins
      if max_value_shift == "default":  max_value_shift = (min_value - max_value) / n_bins
      rad_bdr   = np.linspace( min_value+min_value_shift , max_value+max_value_shift , num=n_bins )
      indices   = [ np.argmax( self.mean_vector >= bdr ) for bdr in rad_bdr ]
      unique_indices = np.unique(indices)
      if len(indices) != len(unique_indices):
        print("WARNING: Some bins were duplicate. These duplicates are removed from the list.")
      self.bins = [ self.bins[i] for i in unique_indices ]
    elif choose_type == "uniform_x":
      max_index = np.amax( np.argmin(self.mean_vector) )
      min_index = np.amin( np.argmax(self.mean_vector) )
      if min_value_shift == "default":  min_value_shift = (max_index - min_index) / n_bins
      if max_value_shift == "default":  max_value_shift = (min_index - max_index) / n_bins
      indices   = np.linspace( min_index+min_value_shift , max_index+max_value_shift , num=n_bins )
      unique_indices = np.unique(indices)
      if len(indices) != len(unique_indices):
        print("WARNING: Some bins were duplicate. These duplicates are removed from the list.")
      self.bins = [ self.bins[int(i)] for i in unique_indices ]
    else:
      print("WARNING: Invalid choose_type flag for choose_bins. Nothing is done in this function.")
      return

    self.ecdf_list = empirical_cumulative_distribution_vector_list(
      self.dataset, self.bins, self.distance_fct, self.subset_indices )
    self.mean_vector   = mean_of_ecdf_vectors(self.ecdf_list)
    self.covar_matrix  = covariance_of_ecdf_vectors(self.ecdf_list)
    if file_output:
      np.savetxt('choose-bins_bins.txt', self.bins, fmt='%.6f')
      np.savetxt('choose-bins_ecdf-list.txt', self.ecdf_list, fmt='%.6f')
      np.savetxt('choose-bins_mean-vector.txt', self.mean_vector, fmt='%.6f')
      np.savetxt('choose-bins_covar-matrix.txt', self.covar_matrix, fmt='%.6f')
    if check_spectral_conditon:
      spectral_condition = np.linalg.cond(self.covar_matrix)
      if spectral_condition > 1e3:
        print("WARNING: The spectral condition of the covariance matrix is", spectral_condition)

  def evaluate_from_empirical_cumulative_distribution_functions( self, vector ):
    mean_deviation = np.subtract( self.mean_vector , vector )
    try:
      return np.dot( mean_deviation , np.linalg.solve(self.covar_matrix, mean_deviation) )
    except np.linalg.LinAlgError as error:
      if not self.error_printed:
        self.error_printed = True
        print("WARNING: Covariance matrix is singular. CIL_estimator uses different topology.")
      return np.dot( mean_deviation , mean_deviation )

  def evaluate( self, dataset ):
    comparison_set = np.random.randint( len(self.subset_indices)-1 )
    y = empirical_cumulative_distribution_vector(self.dataset,dataset,self.bins,self.distance_fct,
      self.subset_indices[comparison_set], self.subset_indices[comparison_set+1] )
    return self.evaluate_from_empirical_cumulative_distribution_functions( y )
