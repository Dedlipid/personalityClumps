import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_pca(data, labels, title, filename, n_components=3):
  """Plots PCA results in 2D or 3D based on the number of components with enhanced depth perception."""

  if n_components not in [2, 3]:
      raise ValueError("n_components must be 2 or 3 for plotting.")

  plt.figure(figsize=(8, 6) if n_components == 2 else (10, 8))

  # Calculate distances from the origin
  distances = np.linalg.norm(data, axis=1)
  # Normalize distances to range [0, 1]
  normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())

  if n_components == 2:
      for i, label in enumerate(labels):
          plt.scatter(data[i, 0], data[i, 1], alpha=normalized_distances[i])
          plt.text(data[i, 0], data[i, 1], label, fontsize=9, ha="right")
      plt.xlabel("Principal Component 1")
      plt.ylabel("Principal Component 2")
      plt.title(title)
      plt.savefig(f"{filename}.png")
      plt.close()
  elif n_components == 3:
      # Create a Plotly 3D scatter plot
      fig = go.Figure()

      # Define colors for each octant
      octant_colors = {
          (1, 1, 1): "red",
          (1, 1, -1): "green",
          (1, -1, 1): "blue",
          (1, -1, -1): "cyan",
          (-1, 1, 1): "magenta",
          (-1, 1, -1): "yellow",
          (-1, -1, 1): "orange",
          (-1, -1, -1): "purple",
      }

      for i, label in enumerate(labels):
          # Determine the octant
          octant = (np.sign(data[i, 0]), np.sign(data[i, 1]), np.sign(data[i, 2]))
          color = octant_colors[octant]

          # Set a constant size for all points
          size = 10  # Constant size for all points

          # Add the main point
          fig.add_trace(go.Scatter3d(
              x=[data[i, 0]], y=[data[i, 1]], z=[data[i, 2]],
              mode='markers+text',
              marker=dict(size=size, color=color, opacity=0.8),
              text=[label],
              textposition="top center"
          ))

      # Calculate the axis length
      max_range = np.max(np.abs(data), axis=0) * 1.1  # Extend a bit beyond data range

      # Add lines for the x, y, and z axes in both directions
      fig.add_trace(go.Scatter3d(
          x=[-max_range[0], max_range[0]], y=[0, 0], z=[0, 0],
          mode='lines',
          line=dict(color='black', width=2),
          name='X-axis'
      ))
      fig.add_trace(go.Scatter3d(
          x=[0, 0], y=[-max_range[1], max_range[1]], z=[0, 0],
          mode='lines',
          line=dict(color='black', width=2),
          name='Y-axis'
      ))
      fig.add_trace(go.Scatter3d(
          x=[0, 0], y=[0, 0], z=[-max_range[2], max_range[2]],
          mode='lines',
          line=dict(color='black', width=2),
          name='Z-axis'
      ))

      # Set axis labels
      fig.update_layout(
          scene=dict(
              xaxis=dict(title='Principal Component 1', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white"),
              yaxis=dict(title='Principal Component 2', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white"),
              zaxis=dict(title='Principal Component 3', backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white")
          ),
          title=title
      )

      # Save as an interactive HTML file
      fig.write_html(f"{filename}.html")
      fig.show()

# Example usage:
# plot_pca(data, labels, "PCA Plot", "pca_plot", n_components=3)