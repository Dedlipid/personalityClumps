import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from raw_data import names
from general_data import dimension_names
from personality_profile import PersonalityProfile
from general_data import dimension_names, overall_indices, no_overall_indices


def plot_top_variance_components(data, labels, dimension_names, title, filename):
  """Plots the data on the three components with the highest variance."""
  # Extract overall scores
  overall_data = data[:, overall_indices]

  # Perform PCA
  pca = PCA()
  pca.fit(overall_data)

  # Get the indices of the top 3 components with the highest variance
  top_indices = np.argsort(pca.explained_variance_)[-3:][::-1]

  # Transform the data using these components
  transformed_data = pca.transform(overall_data)[:, top_indices]

  # Plot the data
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')
  for i, label in enumerate(labels):
      ax.scatter(transformed_data[i, 0], transformed_data[i, 1], transformed_data[i, 2])
      ax.text(transformed_data[i, 0], transformed_data[i, 1], transformed_data[i, 2], label, fontsize=9, ha='right')

  ax.set_xlabel(f'Component {top_indices[0] + 1}')
  ax.set_ylabel(f'Component {top_indices[1] + 1}')
  ax.set_zlabel(f'Component {top_indices[2] + 1}')
  plt.title(title)
  plt.grid(True)
  plt.savefig(filename)
  plt.close()

# Example usage:
# Assuming `data` is your dataset and `labels` are the labels for each data point
# dimension_names = [...]  # List of dimension names
# plot_top_variance_components(data, labels, dimension_names, "Top 3 Variance Components", "top_variance_components.png")

def main():
    profiles = {key: PersonalityProfile(value).overall_to_np_array() for key, value in names.items()}

    # Load data
    data = profiles.values()

    labels = list(profiles.keys())
    # Indices for overall scores
    overall_indices = [i for i, name in enumerate(dimension_names) if "overall" in name]



    # Plot the top 3 variance components
    plot_top_variance_components(data, labels, dimension_names, "Top 3 Variance Components", "top_variance_components.png")