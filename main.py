import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from personality_profile import PersonalityProfile
from raw_data import names


def extract_features(profile :PersonalityProfile) -> list:
  """Extracts features from a personality profile for PCA."""
  features = []
  for trait, values in profile.items():
      features.append(values['overall'])
      features.extend(values['sub'].values())
  return features

def pca_analysis(data, dimension_names, n_components=3, indices=None, print_components=False):
  """Performs PCA and sorts components by magnitude."""
  if indices is not None:
      data = data[:, indices]
      dimension_names = [dimension_names[i] for i in indices]
  
  pca = PCA(n_components=n_components)
  transformed_data = pca.fit_transform(data)
  sorted_components = []
  for component in pca.components_:
      component_dict = {dim_name: dim_value for dim_name, dim_value in zip(dimension_names, component)}
      sorted_component = dict(sorted(component_dict.items(), key=lambda item: abs(item[1]), reverse=True))
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
          plt.text(data[i, 0], data[i, 1], label, fontsize=9, ha='right')
      plt.xlabel('Principal Component 1')
      plt.ylabel('Principal Component 2')
  elif n_components == 3:
      ax = plt.axes(projection='3d')
      for i, label in enumerate(labels):
          ax.scatter(data[i, 0], data[i, 1], data[i, 2])
          ax.text(data[i, 0], data[i, 1], data[i, 2], label, fontsize=9, ha='right')
      ax.set_xlabel('Principal Component 1')
      ax.set_ylabel('Principal Component 2')
      ax.set_zlabel('Principal Component 3')
  
  plt.title(title)
  plt.grid(True)
  plt.savefig(filename)
  plt.close()

def perform_pca_and_plot(data, dimension_names, labels, indices, n_components, title_suffix, filename_suffix):
  """Helper function to perform PCA and plot the results."""
  transformed_data, _ = pca_analysis(data, dimension_names, n_components, indices)
  plot_pca(transformed_data, labels, f'{n_components}D PCA of Personality Profiles ({title_suffix})', f'{n_components}d_pca_{filename_suffix}.png', n_components)

def main():
  profiles = {
      key: PersonalityProfile(value) for key, value in names.items()
  }

  # Extract features for PCA
  data = np.array([extract_features(profile.profile) for profile in profiles.values()])
  labels = list(profiles.keys())

  # Define dimension names
  dimension_names = [
      "agreeableness_overall", "altruism", "cooperation", "modesty", "morality", "sympathy", "trust",
      "conscientiousness_overall", "achievement_striving", "cautiousness", "dutifulness", "orderliness", "self_discipline", "self_efficacy",
      "extraversion_overall", "activity_level", "assertiveness", "cheerfulness", "excitement_seeking", "friendliness", "gregariousness",
      "neuroticism_overall", "anger", "anxiety", "depression", "immoderation", "self_consciousness", "vulnerability",
      "openness_overall", "adventurousness", "artistic_interests", "emotionality", "imagination", "intellect", "liberalism"
  ]

  # Indices for overall scores
  overall_indices = [i for i, name in enumerate(dimension_names) if 'overall' in name]
  # Indices excluding overall scores
  no_overall_indices = [i for i in range(len(dimension_names)) if i not in overall_indices]

  # Perform PCA and plot for each scenario
  perform_pca_and_plot(data, dimension_names, labels, None, 2, 'Full Vector', 'full_vector')
  perform_pca_and_plot(data, dimension_names, labels, overall_indices, 2, 'Overall Scores Only', 'overall_scores')
  perform_pca_and_plot(data, dimension_names, labels, no_overall_indices, 2, 'Excluding Overall', 'no_overall')

  perform_pca_and_plot(data, dimension_names, labels, None, 3, 'Full Vector', 'full_vector')
  perform_pca_and_plot(data, dimension_names, labels, overall_indices, 3, 'Overall Scores Only', 'overall_scores')
  perform_pca_and_plot(data, dimension_names, labels, no_overall_indices, 3, 'Excluding Overall', 'no_overall')

if __name__ == "__main__":
  main()