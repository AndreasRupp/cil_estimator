import numpy as np
import ecdf_estimator.utils as ecdf_aux


## \brief  Objective function assembled in the stardard ecdf way.
class standard:
  ## \brief  Construct objective function.
  def __init__( self, dataset, bins, distance_fct, subset_sizes, compare_all=True ):
    if len(np.unique(subset_sizes)) > 2:
      raise Exception("ERROR: There should be max 2 different sizes of subsets!")
    
    self.dataset        = dataset
    self.bins           = bins
    self.distance_fct   = distance_fct
    self.subset_sizes   = subset_sizes
    self.subset_indices = [ sum(subset_sizes[:i]) for i in range(len(subset_sizes)+1) ]
    self.compare_all    = compare_all
    self.ecdf_list      = ecdf_aux.empirical_cumulative_distribution_vector_list(
                            dataset, bins, distance_fct, self.subset_indices, compare_all )
    self.mean_vector    = ecdf_aux.mean_of_ecdf_vectors(self.ecdf_list)
    self.covar_matrix   = ecdf_aux.covariance_of_ecdf_vectors(self.ecdf_list)
    self.error_printed  = False

  def evaluate_ecdf(self, dataset):
    if len(dataset) not in self.subset_sizes:
      print("WARNING: Dataset size is different!")
    comparison_set = np.random.randint( len(self.subset_sizes) )
    while not self.compare_all and self.subset_sizes[i] == len(dataset):
      comparison_set = np.random.randint( len(self.subset_sizes) )
    distance_list = ecdf_aux.create_distance_matrix(self.dataset, dataset,
      self.distance_fct, self.subset_indices[comparison_set], self.subset_indices[comparison_set+1])
    while isinstance(distance_list[0], list):
      distance_list = [item for sublist in distance_list for item in sublist]
    return ecdf_aux.empirical_cumulative_distribution_vector(distance_list, self.bins)

  def evaluate( self, dataset ):
    return ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions( self,
      self.evaluate_ecdf(dataset) )


## \brief  Objective function assembled via bootstrapping.
class bootstrap:
  ## \brief  Construct objective function.
  def __init__( self, dataset, bins, distance_fct, n_elements_a, n_eleemnts_b, n_samples=1000 ):
    self.dataset        = dataset
    self.bins           = bins
    self.distance_fct   = distance_fct
    self.n_elements_a   = n_elements_a
    self.n_eleemnts_b   = n_eleemnts_b
    self.n_samples      = n_samples
    self.ecdf_list      = ecdf_aux.empirical_cumulative_distribution_vector_list_bootstrap(
                            dataset, bins, distance_fct, n_elements_a, n_elements_b, n_samples )
    self.mean_vector    = ecdf_aux.mean_of_ecdf_vectors(self.ecdf_list)
    self.covar_matrix   = ecdf_aux.covariance_of_ecdf_vectors(self.ecdf_list)
    self.error_printed  = False

  def evaluate_ecdf( self, dataset ):
    if len(dataset) != n_elements_a:
      print("WARNING: The size of the dataset should be equal to n_elements_a!")
    
    samples = np.random.ranint(len(dataset), size=n_elements_b)
    comparison_set = dataset[samples]

    distance_list = ecdf_aux.create_distance_matrix(comparison_set, dataset, self.distance_fct)
    distance_list = [item for sublist in distance_list for item in sublist]
    return ecdf_aux.empirical_cumulative_distribution_vector(distance_list, self.bins)

  def evaluate( self, dataset ):
    return ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions( self,
      self.evaluate_ecdf(dataset) )


## \brief  Objective function that consists of mutliple objective functions.
class multiple:
  ## \brief  Construct objective function.
  def __init__( self, obj_fun_list ):
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

  def evaluate( self, dataset ):
    vector = [ obj_fun.evaluate_ecdf(dataset) for obj_fun in self.obj_fun_list ]
    while isinstance(vector[0], list):
      vector = [item for sublist in vector for item in sublist]
    return ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions( self, vector )
