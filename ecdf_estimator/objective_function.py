import numpy as np
import ecdf_estimator.auxiliaries as ecdf_aux


# --------------------------------------------------------------------------------------------------
class objective_function:
  def __init__( self, dataset, bins, distance_fct, subset_sizes, file_output = False ):
    self.type           = "standard"
    self.dataset        = dataset
    self.bins           = bins
    self.distance_fct   = distance_fct
    self.subset_indices = [ sum(subset_sizes[:i]) for i in range(len(subset_sizes)+1) ]
    self.ecdf_list      = ecdf_aux.empirical_cumulative_distribution_vector_list(
                            dataset, bins, distance_fct, self.subset_indices )
    self.mean_vector    = ecdf_aux.mean_of_ecdf_vectors(self.ecdf_list)
    self.covar_matrix   = ecdf_aux.covariance_of_ecdf_vectors(self.ecdf_list)
    self.error_printed  = False
    if file_output:       ecdf_aux.file_output( self )

  def choose_bins( self, n_bins = 10, min_value_shift = "default", max_value_shift = "default",
    choose_type = "uniform_y", check_spectral_conditon = True, file_output = False ):
    ecdf_aux.choose_bins(self, n_bins, min_value_shift, max_value_shift, choose_type,
      check_spectral_conditon, file_output)

  def evaluate_from_empirical_cumulative_distribution_functions( self, vector ):
    return ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions( self, vector )

  def evaluate_ecdf(self, dataset):
    comparison_set = np.random.randint( len(self.subset_indices)-1 )
    distance_list = ecdf_aux.create_distance_matrix(self.dataset, dataset,
      self.distance_fct, self.subset_indices[comparison_set],
      self.subset_indices[comparison_set+1])
    while isinstance(distance_list[0], list):
      distance_list = [item for sublist in distance_list for item in sublist]
    return ecdf_aux.empirical_cumulative_distribution_vector(distance_list, self.bins)

  def evaluate( self, dataset ):
    return self.evaluate_from_empirical_cumulative_distribution_functions(
      self.evaluate_ecdf(dataset) )

# --------------------------------------------------------------------------------------------------
class bootstrap_objective_function:
  def __init__( self, dataset_a, dataset_b, bins, distance_fct, n_samples = 1000,
    file_output = False ):
    self.type           = "bootstrap"
    self.dataset_a      = dataset_a
    self.dataset_b      = dataset_b
    self.bins           = bins
    self.distance_fct   = distance_fct
    self.n_samples      = n_samples
    self.ecdf_list      = ecdf_aux.empirical_cumulative_distribution_vector_list_bootstrap(
                            dataset_a, dataset_b, bins, distance_fct, self.n_samples )
    self.mean_vector    = ecdf_aux.mean_of_ecdf_vectors(self.ecdf_list)
    self.covar_matrix   = ecdf_aux.covariance_of_ecdf_vectors(self.ecdf_list)
    self.error_printed  = False
    if file_output:       ecdf_aux.file_output( self )

  def choose_bins( self, n_bins = 10, min_value_shift = "default", max_value_shift = "default",
    choose_type = "uniform_y", check_spectral_conditon = True, file_output = False ):
    ecdf_aux.choose_bins(self, n_bins, min_value_shift, max_value_shift, choose_type,
      check_spectral_conditon, file_output)

  def evaluate_from_empirical_cumulative_distribution_functions( self, vector ):
    return ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions( self, vector )

  def evaluate_ecdf( self, dataset ):
    if np.random.randint( 2 ) == 0:  comparison_set = self.dataset_a
    else:                            comparison_set = self.dataset_b

    distance_list = ecdf_aux.create_distance_matrix(comparison_set, dataset, self.distance_fct)
    distance_list = [item for sublist in distance_list for item in sublist]
    return ecdf_aux.empirical_cumulative_distribution_vector(distance_list, self.bins)

  def evaluate( self, dataset ):
    return self.evaluate_from_empirical_cumulative_distribution_functions(
      self.evaluate_ecdf(dataset) )

# --------------------------------------------------------------------------------------------------
class multiple_objectives:
  def __init__( self, obj_fun_list, check_spectral_conditon = True ):
    self.obj_fun_list = obj_fun_list

    n_rows, n_columns = 0, -1
    for obj_fun in obj_fun_list:
      n_rows += obj_fun.ecdf_list.shape[0]
      if n_columns == -1:
        n_columns = obj_fun.ecdf_list.shape[1]
      elif n_columns != obj_fun.ecdf_list.shape[1]:
        print("ERROR: All objective functions should contain the same number of ecdf vectors.")

    self.ecdf_list = np.zeros( (n_rows, n_columns) )
    index = 0
    for obj_fun in obj_fun_list:
      self.ecdf_list[index:index+obj_fun.ecdf_list.shape[0],:] = obj_fun.ecdf_list
      index = index+obj_fun.ecdf_list.shape[0]

    self.mean_vector    = ecdf_aux.mean_of_ecdf_vectors(self.ecdf_list)
    self.covar_matrix   = ecdf_aux.covariance_of_ecdf_vectors(self.ecdf_list)
    self.error_printed  = False

    if check_spectral_conditon:
      spectral_condition = np.linalg.cond(self.covar_matrix)
      if spectral_condition > 1e3:
        print("WARNING: The spectral condition of the covariance matrix is", spectral_condition)

  def evaluate_from_empirical_cumulative_distribution_functions( self, vector ):
    return ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions( self, vector )

  def evaluate( self, dataset ):
    vector = [ obj_fun.evaluate_ecdf(dataset) for obj_fun in self.obj_fun_list ]
    while isinstance(vector[0], list):
      vector = [item for sublist in vector for item in sublist]
    return self.evaluate_from_empirical_cumulative_distribution_functions( vector )
