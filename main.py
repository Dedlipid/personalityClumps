import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from personality_profile import PersonalityProfile
from raw_data import names
from general_data import dimension_names
from pca_tools import pca_analysis, plot_pca, perform_pca_and_plot

def extract_features(profile: PersonalityProfile) -> list:
    """Extracts features from a personality profile for PCA."""
    features = []
    for trait, values in profile.items():
        features.append(values["overall"])
        features.extend(values["sub"].values())
    return features




def main():
    profiles = {key: PersonalityProfile(value) for key, value in names.items()}

    # Extract features for PCA
    data = np.array(
        [profile.to_np_array() for profile in profiles.values()]
    )
    labels = list(profiles.keys())

    # Indices for overall scores
    overall_indices = [i for i, name in enumerate(dimension_names) if "overall" in name]

    # Indices excluding overall scores
    no_overall_indices = [
        i for i in range(len(dimension_names)) if i not in overall_indices
    ]

    # Perform PCA and plot for each scenario
    perform_pca_and_plot(
        data, dimension_names, labels, None, 2, "Full Vector", "full_vector"
    )
    perform_pca_and_plot(
        data,
        dimension_names,
        labels,
        overall_indices,
        2,
        "Overall Scores Only",
        "overall_scores",
    )
    perform_pca_and_plot(
        data,
        dimension_names,
        labels,
        no_overall_indices,
        2,
        "Excluding Overall",
        "no_overall",
    )

    perform_pca_and_plot(
        data, dimension_names, labels, None, 3, "Full Vector", "full_vector"
    )
    perform_pca_and_plot(
        data,
        dimension_names,
        labels,
        overall_indices,
        3,
        "Overall Scores Only",
        "overall_scores",
    )
    perform_pca_and_plot(
        data,
        dimension_names,
        labels,
        no_overall_indices,
        3,
        "Excluding Overall",
        "no_overall",
    )


if __name__ == "__main__":
    main()
