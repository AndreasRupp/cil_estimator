import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import ecdf_estimator.utils as ecdf_aux


def plot_ecdf_vectors( estimator, plotter="default", plot_options="default" ):
  if plotter      == "default":  plotter = plt
  if plot_options == "default":  plot_options = "b."

  if hasattr(estimator, 'bins'):  bins = estimator.bins
  else:                           bins = range(1, len(estimator.ecdf_list)+1)
  
  for vector in np.transpose(estimator.ecdf_list):
    plotter.plot(bins, vector, plot_options)
  return plotter


def plot_mean_vector( estimator, plotter="default", plot_options="default" ):
  if plotter      == "defa-ult":  plotter = plt
  if plot_options == "default":  plot_options = "g."

  if hasattr(estimator, 'bins'):  bins = estimator.bins
  else:                           bins = range(1, len(estimator.ecdf_list)+1)
  
  plotter.plot(bins, estimator.mean_vector, plot_options)
  return plotter


def plot_chi2_test( estimator, plotter="default", n_bins=[], plot_options="default" ):
  if plotter      == "default":  plotter = plt
  if plot_options == "default":  plot_options = "r-"

  n_logl = [ ecdf_aux.evaluate_from_empirical_cumulative_distribution_functions(estimator, vector) \
             for vector in np.transpose(estimator.ecdf_list) ]

  if n_bins == []:  khi, bins = np.histogram( n_logl )
  else:             khi, bins = np.histogram( n_logl, n_bins )

  khi_n = [ x / sum(khi) / (bins[1] - bins[0]) for x in khi ]
  plotter.hist(bins[:-1], bins, weights=khi_n)
  df = len( estimator.ecdf_list )
  x  = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99,df), 100)
  plotter.plot(x, chi2.pdf(x, df),plot_options, lw=5, alpha=0.6, label='chi2 pdf')
  return plotter


def save_data( estimator, name="ecdf_estimator" ):
  np.savetxt(name + '_bins.txt',         estimator.bins, fmt='%.6f')
  np.savetxt(name + '_ecdf-list.txt',    estimator.ecdf_list, fmt='%.6f')
  np.savetxt(name + '_mean-vector.txt',  estimator.mean_vector, fmt='%.6f')
  np.savetxt(name + '_covar-matrix.txt', estimator.covar_matrix, fmt='%.6f')
