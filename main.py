import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from personality_profile import PersonalityProfile
from raw_data import names

def extract_features(profile):
    """Extracts features from a personality profile for PCA."""
    features = []
    for trait, values in profile.items():
        features.append(values['overall'])
        features.extend(values['sub'].values())
    return features

def main():
    profiles = {
        key: PersonalityProfile(value) for key, value in names.items()
    }

    # Extract features for PCA
    data = np.array([extract_features(profile.profile) for profile in profiles.values()])

    # Perform PCA
    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)

    data_2d = pca_2d.fit_transform(data)
    data_3d = pca_3d.fit_transform(data)

    # Define dimension names
    dimension_names = [
        "agreeableness_overall", "altruism", "cooperation", "modesty", "morality", "sympathy", "trust",
        "conscientiousness_overall", "achievement_striving", "cautiousness", "dutifulness", "orderliness", "self_discipline", "self_efficacy",
        "extraversion_overall", "activity_level", "assertiveness", "cheerfulness", "excitement_seeking", "friendliness", "gregariousness",
        "neuroticism_overall", "anger", "anxiety", "depression", "immoderation", "self_consciousness", "vulnerability",
        "openness_overall", "adventurousness", "artistic_interests", "emotionality", "imagination", "intellect", "liberalism"
    ]

    # Print PCA components with dimension names
    print("2D PCA Components:")
    for i, component in enumerate(pca_2d.components_):
        component_dict = {dim_name: dim_value for dim_name, dim_value in zip(dimension_names, component)}
        print(f"Component {i+1}: {component_dict}")

    print("\n3D PCA Components:")
    for i, component in enumerate(pca_3d.components_):
        component_dict = {dim_name: dim_value for dim_name, dim_value in zip(dimension_names, component)}
        print(f"Component {i+1}: {component_dict}")

    # Plot 2D PCA
    plt.figure(figsize=(8, 6))
    for i, (label, profile) in enumerate(profiles.items()):
        plt.scatter(data_2d[i, 0], data_2d[i, 1])
        plt.text(data_2d[i, 0], data_2d[i, 1], label, fontsize=9, ha='right')
    plt.title('2D PCA of Personality Profiles')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.savefig('2d_pca_personality_profiles.png')  # Save the 2D plot
    plt.close()

    # Plot 3D PCA
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, (label, profile) in enumerate(profiles.items()):
        ax.scatter(data_3d[i, 0], data_3d[i, 1], data_3d[i, 2])
        ax.text(data_3d[i, 0], data_3d[i, 1], data_3d[i, 2], label, fontsize=9, ha='right')
    ax.set_title('3D PCA of Personality Profiles')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.savefig('3d_pca_personality_profiles.png')  # Save the 3D plot
    plt.close()

if __name__ == "__main__":
    main()