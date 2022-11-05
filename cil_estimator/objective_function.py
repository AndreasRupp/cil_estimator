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


def correlation_integral_vector_matrix( dataset, radii, distance_fct, subset_indices ):
  if not all(subset_indices[i] <= subset_indices[i+1] for i in range(len(subset_indices)-1)):
    raise Exception("Subset indices are out of order.")
  if subset_indices[0] != 0 or subset_indices[-1] != len(dataset):
    raise Exception("Not all elements of the dataset are distributed into subsets.")

  matrix = []
  for i in range(len(subset_indices)-1):
    for j in range(i):
      matrix.append( correlation_integral_vector( dataset, dataset, radii, distance_fct,
        subset_indices[i], subset_indices[i+1], subset_indices[j], subset_indices[j+1] ) )
  return np.transpose(matrix)


def mean_of_matrix_of_correlation_vectors( matrix_of_vectors_transposed ):
  return [ np.mean(vector) for vector in matrix_of_vectors_transposed ]


def covariance_of_matrix_of_correlation_vectors( matrix_of_vectors_transposed ):
  return np.cov( matrix_of_vectors_transposed )


class objective_function:
  def __init__( self, dataset, radii, distance_fct, subset_sizes, file_output = False ):
    self.dataset       = dataset
    self.radii         = radii
    self.distance_fct  = distance_fct
    self.subset_indices= [ sum(subset_sizes[:i]) for i in range(len(subset_sizes)+1) ]
    self.correlation_vector_matrix = correlation_integral_vector_matrix(
      dataset, radii, distance_fct, self.subset_indices )
    self.mean_vector   = mean_of_matrix_of_correlation_vectors(self.correlation_vector_matrix)
    self.covar_matrix  = covariance_of_matrix_of_correlation_vectors(self.correlation_vector_matrix)
    self.error_printed = False
    if file_output:
      np.savetxt('obj-func_radii.txt', self.radii, fmt='%.6f')
      np.savetxt('obj-func_correlation_vec_mat.txt', self.correlation_vector_matrix, fmt='%.6f')
      np.savetxt('obj-func_mean_vector.txt', self.mean_vector, fmt='%.6f')
      np.savetxt('obj-func_covar_matrix.txt', self.covar_matrix, fmt='%.6f')

  def choose_radii( self, n_radii = 10, min_value_shift = "default", max_value_shift = "default",
    check_spectral_conditon = True, file_output = False ):
    max_value = np.amax( self.mean_vector )
    min_value = np.amin( self.mean_vector )
    if min_value_shift == "default":  min_value_shift = (max_value - min_value) / n_radii
    if max_value_shift == "default":  max_value_shift = (min_value - max_value) / n_radii
    
    rad_bdr   = np.linspace( min_value+min_value_shift , max_value+max_value_shift , num=n_radii )
    indices   = [ np.argmax( self.mean_vector >= bdr ) for bdr in rad_bdr ]
    self.radii         = [ self.radii[i] for i in indices ]
    self.correlation_vector_matrix = correlation_integral_vector_matrix(
      self.dataset, self.radii, self.distance_fct, self.subset_indices )
    self.mean_vector   = mean_of_matrix_of_correlation_vectors(self.correlation_vector_matrix)
    self.covar_matrix  = covariance_of_matrix_of_correlation_vectors(self.correlation_vector_matrix)
    spectral_condition = np.linalg.cond(self.covar_matrix)
    if file_output:
      np.savetxt('choose-radii_radii.txt', self.radii, fmt='%.6f')
      np.savetxt('choose-radii_correlation_vec_mat.txt', self.correlation_vector_matrix, fmt='%.6f')
      np.savetxt('choose-radii_mean_vector.txt', self.mean_vector, fmt='%.6f')
      np.savetxt('choose-radii_covar_matrix.txt', self.covar_matrix, fmt='%.6f')
    if spectral_condition > 1e3:
      print("WARNING: The spectral condition of the covariance matrix is", spectral_condition)

  def evaluate( self, dataset ):
    comparison_set = np.random.randint( len(self.subset_indices)-1 )

    y = correlation_integral_vector( self.dataset, dataset, self.radii, self.distance_fct,
      self.subset_indices[comparison_set], self.subset_indices[comparison_set+1] )
    mean_deviation = np.subtract( self.mean_vector , y )

    try:
      return np.dot( mean_deviation , np.linalg.solve(self.covar_matrix, mean_deviation) )
    except np.linalg.LinAlgError as error:
      if not self.error_printed:
        self.error_printed = True
        print("WARNING: Covariance matrix is singular. CIL_estimator uses different topology.")
      return np.dot( mean_deviation, mean_deviation )
