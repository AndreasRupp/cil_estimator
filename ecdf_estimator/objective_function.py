import numpy as np
from inspect import signature
import ecdf_estimator.utils as ecdf_aux


# --------------------------------------------------------------------------------------------------
# Standard objective function:
# --------------------------------------------------------------------------------------------------

## \brief  Objective function assembled in the stardard ecdf way.
class standard:
  ## \brief  Construct objective function.
  def __init__( self, dataset, bins, distance_fct, subset_sizes, compare_all=True ):
    if len(np.unique(subset_sizes)) > 2:
      raise Exception("ERROR: There should be max 2 different sizes of subsets!")
    ## \private
    self.dataset        = dataset
    ## \private
    self.bins           = bins
    ## \private
    self.distance_fct   = distance_fct
    ## \private
    self.subset_sizes   = subset_sizes
    ## \private
    self.subset_indices = [ sum(subset_sizes[:i]) for i in range(len(subset_sizes)+1) ]
    ## \private
    self.compare_all    = compare_all
    ## \private
    self.ecdf_list      = ecdf_aux.empirical_cumulative_distribution_vector_list(
                            dataset, bins, distance_fct, self.subset_indices, compare_all )
    ## \private
    self.mean_vector    = ecdf_aux.mean_of_ecdf_vectors(self.ecdf_list)
    ## \private
    self.covar_matrix   = ecdf_aux.covariance_of_ecdf_vectors(self.ecdf_list)
    ## \private
    self.error_printed  = False

  ## \private
  def evaluate_ecdf(self, dataset):
    if len(dataset) not in self.subset_sizes:
      print("WARNING: Dataset size is different!")
    
    n_params = len(signature(self.distance_fct).parameters)
    comparison_ind = []
    if self.compare_all:
      for _ in range(n_params-1):
        helper = np.random.randint( len(self.subset_sizes) )
        while helper in comparison_ind:
          helper = np.random.randint( len(self.subset_sizes) )
        comparison_ind.append(helper)
    else:
      for _ in range(n_params-1):
        helper = helper_ind[np.random.randint(len(helper_ind))]
        while helper in comparison_ind:
          helper = np.random.randint( len(self.subset_sizes) )
        comparison_ind.append(helper)

    dataset_list = [dataset] + [self.dataset] * (n_params-1)
    start_index_list = [0] + [ self.subset_indices[index] for index in comparison_ind ]
    end_index_list = [len(dataset)] + [ self.subset_indices[index+1] for index in comparison_ind ]

    distance_list = ecdf_aux.create_distance_matrix(
      dataset_list, self.distance_fct, start_index_list, end_index_list )

    while isinstance(distance_list[0], list):
      distance_list = [item for sublist in distance_list for item in sublist]

    return ecdf_aux.empirical_cumulative_distribution_vector(distance_list, self.bins)

  ## \private
  def evaluate( self, dataset ):
    return ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions(
      self, self.evaluate_ecdf(dataset) )

# --------------------------------------------------------------------------------------------------
# Objective function for bootstrapping:
# --------------------------------------------------------------------------------------------------

## \brief  Objective function assembled via bootstrapping.
class bootstrap:
  ## \brief  Construct objective function.
  def __init__( self, dataset, bins, distance_fct, n_elements_a, n_elements_b, n_samples=1000 ):
    ## \private
    self.dataset        = dataset
    ## \private
    self.bins           = bins
    ## \private
    self.distance_fct   = distance_fct
    ## \private
    self.n_elements_a   = n_elements_a
    ## \private
    self.n_elements_b   = n_elements_b
    ## \private
    self.n_samples      = n_samples
    ## \private
    self.ecdf_list      = ecdf_aux.empirical_cumulative_distribution_vector_list_bootstrap(
                            dataset, bins, distance_fct, n_elements_a, n_elements_b, n_samples )
    ## \private
    self.mean_vector    = ecdf_aux.mean_of_ecdf_vectors(self.ecdf_list)
    ## \private
    self.covar_matrix   = ecdf_aux.covariance_of_ecdf_vectors(self.ecdf_list)
    ## \private
    self.error_printed  = False

  ## \private
  def evaluate_ecdf( self, dataset ):
    if len(dataset) != self.n_elements_a:
      print("WARNING: Size of the dataset should equal n_elements_a!")
    
    comparison_set = [ dataset[i] for i in np.random.randint(len(dataset), size=self.n_elements_b) ]
    dataset_list = [dataset, comparison_set]
    distance_list  = ecdf_aux.create_distance_matrix(dataset_list, self.distance_fct)
    return ecdf_aux.empirical_cumulative_distribution_vector(distance_list, self.bins)

  ## \private
  def evaluate( self, dataset ):
    return ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions( self,
      self.evaluate_ecdf(dataset) )

# --------------------------------------------------------------------------------------------------
# Container class for multiple objective functions:
# --------------------------------------------------------------------------------------------------

## \brief  Objective function that consists of mutliple objective functions.
class multiple:
  ## \brief  Construct objective function.
  def __init__( self, obj_fun_list ):
    ## \private
    self.obj_fun_list = obj_fun_list

    n_rows, n_columns = 0, -1
    for obj_fun in obj_fun_list:
      n_rows += obj_fun.ecdf_list.shape[0]
      if n_columns == -1:
        n_columns = obj_fun.ecdf_list.shape[1]
      elif n_columns != obj_fun.ecdf_list.shape[1]:
        print("ERROR: All objective functions should contain the same number of ecdf vectors.")

    ## \private
    self.ecdf_list = np.zeros( (n_rows, n_columns) )
    index = 0
    for obj_fun in obj_fun_list:
      self.ecdf_list[index:index+obj_fun.ecdf_list.shape[0],:] = obj_fun.ecdf_list
      index = index+obj_fun.ecdf_list.shape[0]
    ## \private
    self.mean_vector    = ecdf_aux.mean_of_ecdf_vectors(self.ecdf_list)
    ## \private
    self.covar_matrix   = ecdf_aux.covariance_of_ecdf_vectors(self.ecdf_list)
    ## \private
    self.error_printed  = False

  ## \private
  def evaluate( self, dataset ):
    vector = [ obj_fun.evaluate_ecdf(dataset[i]) for i,obj_fun in enumerate(self.obj_fun_list) ]
    while isinstance(vector[0], list):
      vector = [item for sublist in vector for item in sublist]
    return ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions( self, vector )
