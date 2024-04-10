import numpy as np
from inspect import signature
import itertools as it
import collections.abc


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

## \brief   Assemble list of (generalized) distances between elemenst of datasets.
#
#  This function creates a list of distances between elements of subsets which are compared to 
#  each other. The amount of subsets compared to each other is defined by the number of arguments
#  in the distance function. The distances of all value combinations of these subsets are computed. 
#
#  \param   dataset_list       The list of datasets which are compared to each other.
#  \param   distance_fct       Function generating a generalized distance between members of datasets.
#  \param   start_index_list   The start indexes to choose datasets which are compared to each other. Defaults to None.
#  \param   end_index_list     The end indexes to choose datasets which are compared to each other. Defaults to None.
#  \retval  distance_mat       Matrix of generalized distances.
def create_distance_matrix(dataset_list, distance_fct, start_index_list=None, end_index_list=None):

  if start_index_list is None:  start_index_list = [0] * len(dataset_list)

  if end_index_list is None:    
    if isinstance(dataset_list[0], collections.abc.Sequence):
      end_index_list = [ len(item) for item in dataset_list ]
    else:
      end_index_list = [0] * len(dataset_list)  

  n_params = len(signature(distance_fct).parameters)

  if ( n_params != 2 or len(dataset_list) != 1 )  and  ( len(dataset_list) != n_params or \
    len(start_index_list) != n_params or len(end_index_list) != n_params ):
    raise Exception("Length of dataset list must be equal to the length of the start index list, "+\
      "the length of the end index list, and the number of arguments of the distance function. "+\
      "\nAlternatively, you can use a single dataset with a binary distance function.")
  
  for k in range(0,len(start_index_list)):
    if end_index_list[k] < start_index_list[k]: raise Exception("Invalid subset indices chosen.")

  if n_params == 1:
    return [distance_fct(dataset_list[0][i]) for i in range(start_index_list[0], end_index_list[0])]

  return [ distance_fct(*item) for item in it.product(*dataset_list)]



## \brief   Assemble ecdf vector, whose elements are list of values for all subset combinations.
#
#  This function assembles a list of ecdf vectors for all possible combinations of subsets of the
#  dataset. Importantly, none of the subsets are compared to themselves. The subset combinations and
#  their indexes are chosen. Then, the distances are computed by create_distance_matrix function
#  and added to a matrix. 
#
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
      distance_list = create_distance_matrix([dataset], distance_fct, [subset_indices[i]], [subset_indices[i+1]])
      
      while isinstance(distance_list[0], list):
        distance_list = [item for sublist in distance_list for item in sublist]
      matrix.append( empirical_cumulative_distribution_vector(distance_list, bins) )

    return np.transpose(matrix)
  # end: if n_params == 1

  combinations = it.combinations(list(range(0,len(subset_indices)-1)), n_params)

  dataset_list = []

  for combo in combinations:
    dataset_list = [dataset[subset_indices[combo[i]]:subset_indices[combo[i]+1]] for i in range(len(combo))]
    start_index_list = [subset_indices[index] for index in combo]
    end_index_list = [subset_indices[index+1] for index in combo]

    distance_list = create_distance_matrix(dataset_list, distance_fct, start_index_list, end_index_list)
    matrix.append(empirical_cumulative_distribution_vector(distance_list, bins))
    dataset_list = []

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
  

  n_params = len(signature(distance_fct).parameters)
  dataset_list = []

  if n_params == 2:
    dataset_list.append(dataset[0:n_elements_a])
    dataset_list.append(dataset[n_elements_a:n_elements_a+n_elements_b])


    distance_matrix = np.array( create_distance_matrix(dataset_list, distance_fct) )

    matrix = []
    for _ in range(n_samples):
      select_a = np.random.randint(len(distance_matrix), size=n_elements_a)
      indices  = [ i for i in range(len(distance_matrix)) if i not in select_a ]
      select_b = [ indices[x] for x in np.random.randint(len(indices), size=n_elements_b) ]


      distance_list = [distance_matrix[i] for sublist in [select_a, select_b] for i in sublist]
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
      x = np.linalg.solve(estimator.covar_matrix, mean_deviation)
      return np.dot( mean_deviation , np.linalg.solve(estimator.covar_matrix, mean_deviation) )
    except np.linalg.LinAlgError as error:
      estimator.error_printed = True
      print("WARNING: Covariance matrix is singular. CIL_estimator uses different topology.")
  return np.dot( mean_deviation , mean_deviation )
