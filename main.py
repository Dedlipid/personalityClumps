# main.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from personality_profile import PersonalityProfile
from raw_data import mrh, atm, hug3

def extract_features(profile):
  """Extracts features from a personality profile for PCA."""
  features = []
  for trait, values in profile.items():
      features.append(values['overall'])
      features.extend(values['sub'].values())
  return features

def main():
  # Create personality profiles
  profiles = {
      "mrh": PersonalityProfile(mrh),
      "atm": PersonalityProfile(atm),
      "hug3": PersonalityProfile(hug3)
  }

  # Extract features for PCA
  data = np.array([extract_features(profile.profile) for profile in profiles.values()])

  # Perform PCA
  pca_2d = PCA(n_components=2)
  pca_3d = PCA(n_components=3)

  data_2d = pca_2d.fit_transform(data)
  data_3d = pca_3d.fit_transform(data)

  # Plot 2D PCA
  plt.figure(figsize=(8, 6))
  for i, (label, profile) in enumerate(profiles.items()):
      plt.scatter(data_2d[i, 0], data_2d[i, 1], label=label)
  plt.title('2D PCA of Personality Profiles')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.legend()
  plt.grid(True)
  plt.savefig('2d_pca_personality_profiles.png')  # Save the 2D plot
  plt.close()

  # Plot 3D PCA
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')
  for i, (label, profile) in enumerate(profiles.items()):
      ax.scatter(data_3d[i, 0], data_3d[i, 1], data_3d[i, 2], label=label)
  ax.set_title('3D PCA of Personality Profiles')
  ax.set_xlabel('Principal Component 1')
  ax.set_ylabel('Principal Component 2')
  ax.set_zlabel('Principal Component 3')
  ax.legend()
  plt.savefig('3d_pca_personality_profiles.png')  # Save the 3D plot
  plt.close()

if __name__ == "__main__":
  main()