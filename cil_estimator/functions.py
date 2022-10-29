def correlation_integral_vector( dataset_a, dataset_b, radii, distance_fct, 
  start_a = 0, end_a = -1, start_b = 0, end_b = -1 ):
  if end_a == -1:
    end_a = len(dataset_a)
  if end_b == -1:
    end_b = len(dataset_b)

  n_close_elem = [0.] * len(radii)
  for elem_a in dataset_a[start_a:end_a]:
    for elem_b in dataset_b[start_b:end_b]:
      distance = distance_fct(elem_a, elem_b)
      n_close_elem = [ ( n_close_elem[i] + (distance < radii[i]) ) for i in range(len(radii)) ]
  return [ elem / ((end_a - start_a) * (end_b - start_b)) for elem in n_close_elem ]


def matrix_of_correlation_integral_vectors( dataset_a, dataset_b, radii, distance_fct, 
  subset_sizes_a = [], subset_sizes_b = [] ):
  if subset_sizes_a == []:
    subset_sizes_a = [len(dataset_a)]
  if subset_sizes_b == []:
    subset_sizes_b = [len(dataset_b)]

  matrix = [[0.] * len(radii) for x in subset_sizes_a for y in subset_sizes_b]
  
  if sum(subset_sizes_a) != len(dataset_a) or sum(subset_sizes_b) != len(dataset_b):
    return matrix

  indices_a = [ sum(subset_sizes_a[:i]) for i in range(len(subset_sizes_a)+1) ]
  indices_b = [ sum(subset_sizes_b[:i]) for i in range(len(subset_sizes_b)+1) ]

  for i in range(len(subset_sizes_a)):
    for j in range(len(subset_sizes_b)):
      matrix[i * len(subset_sizes_b) + j] = correlation_integral_vector( dataset_a, dataset_b,
        radii, distance_fct, indices_a[i], indices_a[i+1], indices_b[j], indices_b[j+1] )

  return matrix


def mean_of_matrix_of_correlation_vectors( matrix_of_vectors ):
  return [sum(vector) / len(vector) for vector in zip(*matrix_of_vectors)]


def mean_and_covariance_of_matrix_of_correlation_vectors( matrix_of_vectors ):
  mean_vector = mean_of_matrix_of_correlation_vectors( matrix_of_vectors )
  covariance_matrix = [[0.] * len(mean_vector) for x in  mean_vector]
  helper = [[vector[i] - mean_vector[i] \
             for i in range(len(mean_vector))] for vector in matrix_of_vectors]

  for vector in helper:
    for i in range(len(mean_vector)):
      for j in range(i+1):
        covariance_matrix[i][j] += vector[i] * vector[j]
  for i in range(len(mean_vector)):
    for j in range(i+1):
      covariance_matrix[i][j] /= len(matrix_of_vectors)
      covariance_matrix[j][i] = covariance_matrix[i][j]

  return mean_vector, covariance_matrix


class objective_function:
  def __init__(self, dataset, radii, distance_fct, subset_sizes):
    self.dataset      = dataset
    self.radii        = radii
    self.distance_fct = distance_fct
    self.subset_sizes = subset_sizes
    matrix_of_correlation_vectors = matrix_of_correlation_integral_vectors(
      dataset, dataset, radii, distance_fct, subset_sizes, subset_sizes )
    self.mean_vector, self.covariance_matrix = \
      mean_and_covariance_of_matrix_of_correlation_vectors( matrix_of_correlation_vectors )

  def evaluate( self, dataset, subset_sizes = [] ):
    if subset_sizes == []:
      subset_sizes = self.subset_sizes

    matrix_of_correlation_vectors = matrix_of_correlation_integral_vectors(
      dataset, self.dataset, self.radii, self.distance_fct, subset_sizes, self.subset_sizes )
    mean_vector = mean_of_matrix_of_correlation_vectors( matrix_of_correlation_vectors )

    helper = [ mean_vector[i] - self.mean_vector[i] for i in range(len(mean_vector)) ]

    result = 0
    for i in range(len(mean_vector)):
      for j in range(len(mean_vector)):
        result += helper[i] * self.covariance_matrix[i][j] * helper[j]

    return result
