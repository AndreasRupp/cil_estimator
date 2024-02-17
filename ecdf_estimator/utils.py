import numpy as np
from inspect import signature
import itertools as it


## \brief   Create list of ECDF values.
#
#  This function creates the empirical cumulative distribution functions from a list of distances
#  and a list of bin values. That is, each element of the resulting list tells how many elements of
#  the distance list are smaller than the respective bin value.
#
#  \param   distance_list  List of distances to be grouped into the bins.
#  \param   bins           List of bins.
#  \retval  ecdf_list      Resulting list of amout of distances that are smaller than resp. bins.
def empirical_cumulative_distribution_vector( distance_list, bins ):
  return [ np.sum( [distance < basket for distance in distance_list] ) / len(distance_list) \
           for basket in bins ]  # np.sum appears to be much faster than Python's standard sum!

## \brief   Assemble matrix of (generalized) distances between elemenst of datasets.
#
#  This function creates a matrix whose (i,j)th entry corresponds to the distance between element
#  i of a subset of dataset_a and element j of a subset of dataset_b. The respective subsets are
#  characterized by the indices of the respective first and last elements.
#  Notably, the matrix entries can be numbers or more general data types (such as lists). If
#  dataset_b is None, all entries of dataset_a in [start_a, end_a] are compared against one another.
#
#  \param   dataset_a      First dataset, whose subset is compared to second dataset.
#  \param   dataset_b      Second dataset, whose subset is compared to first dataset.
#  \param   distance_fct   Function generating a generalized distance between members of datasets.
#  \param   start_a        Starting index of considered subset of dataset_a. Defaults to 0. 
#  \param   end_a          Last (exclusive) index of consideres subset. Defaults to len(dataset).
#  \param   start_b        Starting index of considered subset of dataset_b. Defaults to 0. 
#  \param   end_b          Last (exclusive) index of consideres subset. Defaults to len(dataset).
#  \retval  distance_mat   Matrix of generalized distances.
def create_distance_matrix(dataset_list, distance_fct, start_index_list=None, end_index_list=None):
  if start_index_list is None:  start_index_list = [0] * len(dataset_list)
  if end_index_list is None:    end_index_list = [ len(item) for item in dataset_list ]

  n_params = len(signature(distance_fct).parameters)
  if ( n_params != 2 or len(dataset_list) != 1 )  and  ( len(dataset_list) != n_params or \
    len(start_index_list) != n_params or len(end_index_list) != n_params ):
    raise Exception("Length of dataset list must be equal to the length of the start index list, "+\
      "the length of the end index list, and the number of arguments of the distance function. "+\
      "\nAlternatively, you can use a single dataset with a binary distance function.")
  for k in len(start_index_list):
    if end_index_list[k] < start_index_list[k]: raise Exception("Invalid subset indices chosen.")

  if n_params == 2 and len(dataset_list) == 1:
    if end_index_list[0] != len(dataset_a) or start_index_list[0] != 0:
      raise Exception("You need to use the whole dataset")
    matrix = [ [0.] * (end_index_list[0] - start_index_list[0])
               for _ in range(start_index_list[0], end_index_list[0]) ]
    for i in range(end_a):
      for j in range(i):
        matrix[i][j] = distance_fct(dataset_list[0][i], dataset_list[0][j])
        matrix[j][i] = matrix[i][j]
    return matrix
  # end: if n_params == 2 and len(dataset_list) == 1

  used_data_list = [ dataset[param][start_index_list[param]:end_index_list[param]] \
                     for param in range(n_params) ]

  return [ distance_fct(*item) for item in it.product(*used_data_list) ]


## \brief   Assemble ecdf vector, whose elements are list of values for all subset combinations.
#
#  This function assembles a list of ecdf vectors for all possible combinations of subsets of the
#  dataset. Importantly, none of the subsets are compared to themselves and subsets i and j are
#  compared only once (not i with j and j with i).
#  The first dimension of the result refers to the index of the bin / ecdf vector. The second index
#  of the result refers to the subset combination.
#
#  \param   dataset        First dataset, whose subset are compared to one another.
#  \param   bins           List of bins.
#  \param   distance_fct   Function generating a generalized distance between members of dataset.
#  \param   subset_indices List of starting (and ending) indices of disjointly subdivided dataset.
#  \param   compare_all    If False, only subsets of different sizes are compared. Deafault: True
#  \retval  ecdf_list      ecdf vector enlisting values for subset combinations.
def empirical_cumulative_distribution_vector_list(
  dataset, bins, distance_fct, subset_indices, compare_all=True ):
  n_params = len(signature(distance_fct).parameters)
  matrix = []

  if not all(subset_indices[i] <= subset_indices[i+1] for i in range(len(subset_indices)-1)):
    raise Exception("Subset indices are out of order.")
  if subset_indices[0] != 0 or subset_indices[-1] != len(dataset):
    raise Exception("Not all elements of the dataset are distributed into subsets.")

  if n_params == 1:
    for i in range(len(subset_indices)-1):
      distance_list = create_distance_matrix(dataset, None, distance_fct,
        subset_indices[i], subset_indices[i+1])
      while isinstance(distance_list[0], list):
        distance_list = [item for sublist in distance_list for item in sublist]
      matrix.append( empirical_cumulative_distribution_vector(distance_list, bins) )
    return np.transpose(matrix)
  # end: if n_params == 1

  for i in range(len(subset_indices)-1):
    for j in range(i):
      if not compare_all and \
        subset_indices[i+1] - subset_indices[i] == subset_indices[j+1] - subset_indices[j]:
        continue
      distance_list = create_distance_matrix(dataset, dataset, distance_fct, 
        subset_indices[i], subset_indices[i+1], subset_indices[j], subset_indices[j+1])
      while isinstance(distance_list[0], list):
        distance_list = [item for sublist in distance_list for item in sublist]
      matrix.append( empirical_cumulative_distribution_vector(distance_list, bins) )

  return np.transpose(matrix)


## \brief   Same as empirical_cumulative_distribution_vector_list, but for bootstrapping.
#
#  \param   dataset        Dataset, whose elements are compared to one another.
#  \param   n_elements_a   Number of elements in first (smaller) subset.
#  \param   n_elements_b   Number of elements in second (larger) subset.
#  \param   bins           List of bins.
#  \param   distance_fct   Function generating a generalized distance between members of dataset.
#  \param   n_samples      Number of perturbatins of the datasets.
#  \retval  ecdf_list      ecdf vector enlisting values for subset combinations.
def empirical_cumulative_distribution_vector_list_bootstrap(
  dataset, bins, distance_fct, n_elements_a, n_elements_b, n_samples ):
  distance_matrix = np.array( create_distance_matrix(dataset, None, distance_fct) )
  matrix = []
  for _ in range(n_samples):
    select_a = np.random.randint(len(dataset), size=n_elements_a)
    indices  = [ i for i in range(len(dataset)) if i not in select_a ]
    select_b = [ indices[x] for x in np.random.randint(len(indices), size=n_elements_b) ]

    distance_list = np.ndarray.flatten( distance_matrix[np.ix_(select_a,select_b)] )
    matrix.append( empirical_cumulative_distribution_vector(distance_list, bins) )
  return np.transpose(matrix)


## \brief  Vector of means of the ecdf vectors.
#
#  \param   ecdf_list      Usually the result of empirical_cumulative_distribution_vector_list.
#  \retval  ecdf_means     Element-wise mean values of the ecdf vectors.
def mean_of_ecdf_vectors( ecdf_vector_list ):
  return [ np.mean(vector) for vector in ecdf_vector_list ]

## \brief  Covariance matrix of ecdf vectors.
#
#  \param   ecdf_list      Usually the result of empirical_cumulative_distribution_vector_list.
#  \retval  ecdf_covar     Covariance matrix associated to the ecdf vectors.
def covariance_of_ecdf_vectors( ecdf_vector_list ):
  return np.cov( ecdf_vector_list )


## \brief  Evaluate target/objective/cost function associated to estimator type from dataset.
#
#  Evaluate the negative log-likelihood in the way that is characterized by the estimator.
#
#  \param   estimator      The estimator class defining the specifics of the target function.
#  \param   dataset        The dataset with respect to which the target function is evaluated.
#  \retval  target_val     The value of the target function.
def evaluate( estimator, dataset ):
  return estimator.evaluate( dataset )

## \brief  Evaluate target/objective/cost function associated to estimator type from ecdf vector.
#
#  Evaluate the negative log-likelihood in the way that is characterized by the estimator.
#
#  \param   estimator      The estimator class defining the specifics of the target function.
#  \param   ecdf_vector    The vector of ecdf, which is the argument for the target function.
#  \retval  target_val     The value of the target function.
def evaluate_from_empirical_cumulative_distribution_functions( estimator, vector ):
  mean_deviation = np.subtract( estimator.mean_vector , vector )
  if not estimator.error_printed:
    try:
      return np.dot( mean_deviation , np.linalg.solve(estimator.covar_matrix, mean_deviation) )
    except np.linalg.LinAlgError as error:
      estimator.error_printed = True
      print("WARNING: Covariance matrix is singular. CIL_estimator uses different topology.")
  return np.dot( mean_deviation , mean_deviation )
