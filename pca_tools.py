from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def pca_analysis(
  data, dimension_names, n_components=3, indices=None, print_components=False
):
  """Performs PCA and sorts components by magnitude."""
  if indices is not None:
      data = data[:, indices]
      dimension_names = [dimension_names[i] for i in indices]

  pca = PCA(n_components=n_components)
  transformed_data = pca.fit_transform(data)
  sorted_components = []
  for component in pca.components_:
      component_dict = {
          dim_name: dim_value
          for dim_name, dim_value in zip(dimension_names, component)
      }
      sorted_component = dict(
          sorted(component_dict.items(), key=lambda item: abs(item[1]), reverse=True)
      )
      sorted_components.append({k: f"{v:.2f}" for k, v in sorted_component.items()})

  if print_components:
      for i, component in enumerate(sorted_components):
          print(f"Principal Component {i+1}:")
          for dim, value in component.items():
              print(f"  {dim}: {value}")

  return transformed_data, sorted_components


def plot_pca(data, labels, title, filename, n_components=3):
  """Plots PCA results in 2D or 3D based on the number of components."""
  if n_components not in [2, 3]:
      raise ValueError("n_components must be 2 or 3 for plotting.")

  plt.figure(figsize=(8, 6) if n_components == 2 else (10, 8))

  if n_components == 2:
      for i, label in enumerate(labels):
          plt.scatter(data[i, 0], data[i, 1])
          plt.text(data[i, 0], data[i, 1], label, fontsize=9, ha="right")
      plt.xlabel("Principal Component 1")
      plt.ylabel("Principal Component 2")
  elif n_components == 3:
      ax = plt.axes(projection="3d")
      
      # Determine the limits for the axes
      max_range = np.array([data[:, 0].max() - data[:, 0].min(), 
                            data[:, 1].max() - data[:, 1].min(), 
                            data[:, 2].max() - data[:, 2].min()]).max() / 2.0

      mid_x = (data[:, 0].max() + data[:, 0].min()) * 0.5
      mid_y = (data[:, 1].max() + data[:, 1].min()) * 0.5
      mid_z = (data[:, 2].max() + data[:, 2].min()) * 0.5

      # Define colors for each octant
      octant_colors = {
          (1, 1, 1): 'red',
          (1, 1, -1): 'green',
          (1, -1, 1): 'blue',
          (1, -1, -1): 'cyan',
          (-1, 1, 1): 'magenta',
          (-1, 1, -1): 'yellow',
          (-1, -1, 1): 'orange',
          (-1, -1, -1): 'purple'
      }

      # Plot the main points and their shadows
      for i, label in enumerate(labels):
          # Determine the octant
          octant = (np.sign(data[i, 0]), np.sign(data[i, 1]), np.sign(data[i, 2]))
          color = octant_colors[octant]

          # Plot the main point
          ax.scatter(data[i, 0], data[i, 1], data[i, 2], color=color)
          ax.text(data[i, 0], data[i, 1], data[i, 2], label, fontsize=9, ha="right")
          
          # Plot shadows on the XY plane
          ax.scatter(data[i, 0], data[i, 1], 0, color='gray', alpha=0.5, marker='o')
          # Plot shadows on the XZ plane
          ax.scatter(data[i, 0], 0, data[i, 2], color='gray', alpha=0.5, marker='o')
          # Plot shadows on the YZ plane
          ax.scatter(0, data[i, 1], data[i, 2], color='gray', alpha=0.5, marker='o')

          # Draw lines to shadows
          ax.plot([data[i, 0], data[i, 0]], [data[i, 1], data[i, 1]], [data[i, 2], 0], color='gray', alpha=0.3, linestyle='--')
          ax.plot([data[i, 0], data[i, 0]], [data[i, 1], 0], [data[i, 2], data[i, 2]], color='gray', alpha=0.3, linestyle='--')
          ax.plot([data[i, 0], 0], [data[i, 1], data[i, 1]], [data[i, 2], data[i, 2]], color='gray', alpha=0.3, linestyle='--')

      # Plot the X, Y, and Z axes
      ax.plot([mid_x - max_range, mid_x + max_range], [mid_y, mid_y], [mid_z, mid_z], color='black', alpha=0.7)
      ax.plot([mid_x, mid_x], [mid_y - max_range, mid_y + max_range], [mid_z, mid_z], color='black', alpha=0.7)
      ax.plot([mid_x, mid_x], [mid_y, mid_y], [mid_z - max_range, mid_z + max_range], color='black', alpha=0.7)

      ax.set_xlabel("Principal Component 1")
      ax.set_ylabel("Principal Component 2")
      ax.set_zlabel("Principal Component 3")

  plt.title(title)
  plt.grid(True)
  plt.savefig(filename)
  plt.close()


def perform_pca_and_plot(
  data,
  dimension_names,
  labels,
  indices,
  n_components,
  title_suffix,
  filename_suffix,
  print_components=False,
  n_draw=3,
):
  if n_draw > n_components:
      n_draw = n_components
  """Helper function to perform PCA and plot the results."""
  transformed_data, _ = pca_analysis(
      data, dimension_names, n_components, indices, print_components=print_components
  )
  plot_pca(
      transformed_data,
      labels,
      f"{n_components}D PCA of Personality Profiles ({title_suffix})",
      f"{n_components}d_pca_{filename_suffix}.png",
      n_draw,
  )