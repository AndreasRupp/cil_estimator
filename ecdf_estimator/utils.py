import numpy as np


def empirical_cumulative_distribution_vector( distance_list, bins ):
  return [ np.sum( [distance < basket for distance in distance_list] ) / len(distance_list) \
           for basket in bins ]  # np.sum appears to be much faster than Python's standard sum!


def create_distance_matrix( dataset_a, dataset_b, distance_fct, 
  start_a = 0, end_a = None, start_b = 0, end_b = None ):
  if end_a is None:  end_a = len(dataset_a)
  if end_b is None:  end_b = len(dataset_b)

  return [ [ distance_fct(dataset_a[i], dataset_b[j]) for j in range(start_b, end_b) ] \
             for i in range(start_a, end_a) ]


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
  dataset_a, dataset_b, bins, distance_fct, n_samples ):
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


def evaluate( estimator, dataset ):
  return estimator.evaluate( dataset )


def evaluate_from_empirical_cumulative_distribution_functions( estimator, vector ):
  mean_deviation = np.subtract( estimator.mean_vector , vector )
  try:
    return np.dot( mean_deviation , np.linalg.solve(estimator.covar_matrix, mean_deviation) )
  except np.linalg.LinAlgError as error:
    if not estimator.error_printed:
      estimator.error_printed = True
      print("WARNING: Covariance matrix is singular. CIL_estimator uses different topology.")
    return np.dot( mean_deviation , mean_deviation )
