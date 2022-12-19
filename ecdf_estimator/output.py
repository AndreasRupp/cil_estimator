import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import ecdf_estimator.utils as ecdf_aux


## \brief  Plot all ecdf vectors.
#
#  \param   estimator      The estimator class defining the specifics of the target function.
#  \param   plotter        Python object to which the plot should be added. Defaults to plt.
#  \param   plot_options   String describing the options for plotting.
#  \retval  plotter        Python object to whhich the plot has been added.
def plot_ecdf_vectors( estimator, plotter=plt, plot_options="b." ):
  if hasattr(estimator, 'bins'):  bins = estimator.bins
  else:                           bins = range(1, len(estimator.ecdf_list)+1)
  
  for vector in np.transpose(estimator.ecdf_list):
    plotter.plot(bins, vector, plot_options)
  return plotter


## \brief  Plot means of ecdf vectors.
#
#  \param   estimator      The estimator class defining the specifics of the target function.
#  \param   plotter        Python object to which the plot should be added. Defaults to plt.
#  \param   plot_options   String describing the options for plotting.
#  \retval  plotter        Python object to whhich the plot has been added.
def plot_mean_vector( estimator, plotter=plt, plot_options="g." ):
  if hasattr(estimator, 'bins'):  bins = estimator.bins
  else:                           bins = range(1, len(estimator.ecdf_list)+1)
  
  plotter.plot(bins, estimator.mean_vector, plot_options)
  return plotter


## \brief  Plot chi square test.
#
#  \param   estimator      The estimator class defining the specifics of the target function.
#  \param   plotter        Python object to which the plot should be added. Defaults to plt.
#  \param   plot_options   String describing the options for plotting.
#  \retval  plotter        Python object to whhich the plot has been added.
def plot_chi2_test( estimator, plotter=plt, plot_options="r-" ):
  n_logl = [ ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions(estimator, vector) \
             for vector in np.transpose(estimator.ecdf_list) ]
  khi, bins = np.histogram( n_logl )
  khi_n = [ x / sum(khi) / (bins[1] - bins[0]) for x in khi ]
  plotter.hist(bins[:-1], bins, weights=khi_n)
  df = len( estimator.ecdf_list )
  x  = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99,df), 100)
  plotter.plot(x, chi2.pdf(x, df),plot_options, lw=5, alpha=0.6, label='chi2 pdf')
  return plotter


## \brief  Save ecdf vectors, mean of ecdf vectors, covariance matrix and bin values to files.
#
#  \param   estimator      The estimator class defining the specifics of the target function.
#  \param   name           Prefix of the file names to which data is saved.
def save_data( estimator, name="ecdf_estimator" ):
  np.savetxt(name + '_ecdf-list.txt',    estimator.ecdf_list, fmt='%.6f')
  np.savetxt(name + '_mean-vector.txt',  estimator.mean_vector, fmt='%.6f')
  np.savetxt(name + '_covar-matrix.txt', estimator.covar_matrix, fmt='%.6f')
  if hasattr(estimator, "bins"):  np.savetxt(name + '_bins.txt', estimator.bins, fmt='%.6f')
