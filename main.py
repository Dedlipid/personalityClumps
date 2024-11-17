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

def pca_analysis(data, dimension_names, n_components, indices=None):
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
    return transformed_data, sorted_components

def plot_2d(data, labels, title, filename):
    """Plots 2D PCA results."""
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.scatter(data[i, 0], data[i, 1])
        plt.text(data[i, 0], data[i, 1], label, fontsize=9, ha='right')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_3d(data, labels, title, filename):
    """Plots 3D PCA results."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, label in enumerate(labels):
        ax.scatter(data[i, 0], data[i, 1], data[i, 2])
        ax.text(data[i, 0], data[i, 1], data[i, 2], label, fontsize=9, ha='right')
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.savefig(filename)
    plt.close()

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
    data_2d_full, _ = pca_analysis(data, dimension_names, 2)
    plot_2d(data_2d_full, labels, '2D PCA of Personality Profiles (Full Vector)', '2d_pca_full_vector.png')

    data_2d_overall, _ = pca_analysis(data, dimension_names, 2, overall_indices)
    plot_2d(data_2d_overall, labels, '2D PCA of Personality Profiles (Overall Scores Only)', '2d_pca_overall_scores.png')

    data_2d_no_overall, _ = pca_analysis(data, dimension_names, 2, no_overall_indices)
    plot_2d(data_2d_no_overall, labels, '2D PCA of Personality Profiles (Excluding Overall)', '2d_pca_no_overall.png')

    data_3d_full, _ = pca_analysis(data, dimension_names, 3)
    plot_3d(data_3d_full, labels, '3D PCA of Personality Profiles (Full Vector)', '3d_pca_full_vector.png')

    data_3d_overall, _ = pca_analysis(data, dimension_names, 3, overall_indices)
    plot_3d(data_3d_overall, labels, '3D PCA of Personality Profiles (Overall Scores Only)', '3d_pca_overall_scores.png')

    data_3d_no_overall, _ = pca_analysis(data, dimension_names, 3, no_overall_indices)
    plot_3d(data_3d_no_overall, labels, '3D PCA of Personality Profiles (Excluding Overall)', '3d_pca_no_overall.png')

if __name__ == "__main__":
    main()