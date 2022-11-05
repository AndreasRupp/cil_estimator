import matplotlib.pyplot as plt


def plot_correlation_vectors( objective_function, plotter="default", plot_options="default" ):
  if plotter      == "default":  plotter = plt
  if plot_options == "default":  plot_options = "b."
  for vector in list(zip(*(objective_function.correlation_vector_matrix))):
    plotter.plot(objective_function.radii, vector, plot_options)
  return plotter


def plot_mean_vector( objective_function, plotter="default", plot_options="default" ):
  if plotter      == "default":  plotter = plt
  if plot_options == "default":  plot_options = "g."
  plotter.plot(objective_function.radii, objective_function.mean_vector, plot_options)
  return plotter
