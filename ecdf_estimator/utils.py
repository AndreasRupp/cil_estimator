import numpy as np


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
#  Notably, the matrix entries can be numbers or more general data types (such as lists).
#
#  \param   dataset_a      First dataset, whose subset is compared to second dataset.
#  \param   dataset_b      Second dataset, whose subset is compared to first dataset.
#  \param   distance_fct   Function generating a generalized distance between members of datasets.
#  \param   start_a        Starting index of considered subset of dataset_a. Defaults to 0. 
#  \param   end_a          Last (exclusive) index of consideres subset. Defaults to len(dataset).
#  \param   start_b        Starting index of considered subset of dataset_b. Defaults to 0. 
#  \param   end_b          Last (exclusive) index of consideres subset. Defaults to len(dataset).
#  \retval  distance_mat   Matrix of generalized distances.
def create_distance_matrix( dataset_a, dataset_b, distance_fct, 
  start_a=0, end_a=None, start_b=0, end_b=None ):
  if end_a is None:  end_a = len(dataset_a)

  if dataset_b is None:
    if not end_a == len(dataset_a) or not start_a == 0:
      raise Exception("You need to use the whole dataset")
    matrix = [ [0.] * (end_a - start_a) for _ in range(end_a - start_a) ]
    for i in range(end_a - start_a):
      for j in range(i):
        matrix[i,j] = distance_fct(dataset_a[i], dataset_a[j])
        matrix[j,i] = matrix[i,j]
    return matrix

  if end_b is None:  end_b = len(dataset_b)
  if end_a < start_a or end_b < start_b:
    raise Exception("Invalid subset indices chosen.")

  return [ [ distance_fct(dataset_a[i], dataset_b[j]) for j in range(start_b, end_b) ] \
             for i in range(start_a, end_a) ]


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
#  \retval  ecdf_list      ecdf vector enlisting values for subset combinations.
def empirical_cumulative_distribution_vector_list(
  dataset, bins, distance_fct, subset_indices, compare_all=True ):
  if not all(subset_indices[i] <= subset_indices[i+1] for i in range(len(subset_indices)-1)):
    raise Exception("Subset indices are out of order.")
  if subset_indices[0] != 0 or subset_indices[-1] != len(dataset):
    raise Exception("Not all elements of the dataset are distributed into subsets.")

  matrix = []
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
#  \param   dataset_a      First dataset, whose elements are compared to second dataset.
#  \param   dataset_b      Second dataset, whose elements iare compared to first dataset.
#  \param   bins           List of bins.
#  \param   distance_fct   Function generating a generalized distance between members of dataset.
#  \param   n_samples      Number of perturbatins of the datasets.
#  \retval  ecdf_list      ecdf vector enlisting values for subset combinations.
def empirical_cumulative_distribution_vector_list_bootstrap(
  dataset, bins, distance_fct, n_elements_a, n_elements_b, n_samples ):
  distance_matrix = np.array( create_distance_matrix(dataset, None, distance_fct) )
  matrix = []
  for _ in range(n_samples):
    indices  = [i for i in range(len(dataset))]
    select_a = np.random.ranint(len(dataset), size=n_elements_a)
    indices[select_a] = None
    inidces = [x for x in indices if x is not None]
    select_b = np.random.ranint(len(indices), size=n_elements_b)
    select_b = indices[select_b]

    distance_list = np.ndarray.flatten( distance_matrix[select_a,select_b] )
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
