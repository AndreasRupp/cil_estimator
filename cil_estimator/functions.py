import numpy as np

def correlation_integral_vector( dataset_a, dataset_b, radii, distance_fct, 
  start_a = 0, end_a = -1, start_b = 0, end_b = -1 ):
  if end_a == -1:  end_a = len(dataset_a)
  if end_b == -1:  end_b = len(dataset_b)

  n_close_elem = [0.] * len(radii)            # Use of np.zeros( len(radii) ) slows down code!
  for elem_a in dataset_a[start_a:end_a]:
    for elem_b in dataset_b[start_b:end_b]:
      distance = distance_fct(elem_a, elem_b)
      n_close_elem = [ ( n_close_elem[i] + (distance < radii[i]) ) for i in range(len(radii)) ]
  return [ elem / ((end_a - start_a) * (end_b - start_b)) for elem in n_close_elem ]


def matrix_of_correlation_integral_vectors_transposed( dataset_a, dataset_b, radii, distance_fct, 
  subset_sizes_a = [], subset_sizes_b = [] ):
  if subset_sizes_a == []:  subset_sizes_a = [len(dataset_a)]
  if subset_sizes_b == []:  subset_sizes_b = [len(dataset_b)]
  if sum(subset_sizes_a) != len(dataset_a) or sum(subset_sizes_b) != len(dataset_b):
    raise SizeError("Sizes of subsets do not fit datasets!")

  matrix    = [ [0.] * len(radii) for x in subset_sizes_a for y in subset_sizes_b ]
  indices_a = [ sum(subset_sizes_a[:i]) for i in range(len(subset_sizes_a)+1) ]
  indices_b = [ sum(subset_sizes_b[:i]) for i in range(len(subset_sizes_b)+1) ]

  for i in range(len(subset_sizes_a)):
    for j in range(len(subset_sizes_b)):
      matrix[i * len(subset_sizes_b) + j] = correlation_integral_vector( dataset_a, dataset_b,
        radii, distance_fct, indices_a[i], indices_a[i+1], indices_b[j], indices_b[j+1] )
  return np.transpose(matrix)


def mean_of_matrix_of_correlation_vectors( matrix_of_vectors_transposed ):
  return [ np.mean(vector) for vector in matrix_of_vectors_transposed ]


def covariance_of_matrix_of_correlation_vectors( matrix_of_vectors_transposed ):
  return np.cov( matrix_of_vectors_transposed )


class objective_function:
  def __init__(self, dataset, radii, distance_fct, subset_sizes):
    self.dataset      = dataset
    self.radii        = radii
    self.distance_fct = distance_fct
    self.subset_sizes = subset_sizes
    self.correlation_vector_matrix = matrix_of_correlation_integral_vectors_transposed(
      dataset, dataset, radii, distance_fct, subset_sizes, subset_sizes )
    self.mean_vector  = mean_of_matrix_of_correlation_vectors(self.correlation_vector_matrix)
    self.covar_matrix = covariance_of_matrix_of_correlation_vectors(self.correlation_vector_matrix)

  def correlation_matrix(self):
    return self.correlation_vector_matrix

  def radii(self):
    return self.radii

  def evaluate( self, dataset, subset_sizes = [] ):
    if subset_sizes == []:  subset_sizes = self.subset_sizes

    matrix_of_correlation_vectors = matrix_of_correlation_integral_vectors_transposed(
      dataset, self.dataset, self.radii, self.distance_fct, subset_sizes, self.subset_sizes )
    mean_deviation = np.subtract( self.mean_vector ,
      mean_of_matrix_of_correlation_vectors( matrix_of_correlation_vectors ) )
    return np.dot( mean_deviation , np.dot(self.covar_matrix, mean_deviation) )
