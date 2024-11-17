import numpy as np
from personality_profile import PersonalityProfile
from raw_data import names
from general_data import dimension_names, overall_indices, no_overall_indices
from pca_tools import perform_pca_and_plot

def main():
    # Extract features for PCA
    data = np.array([PersonalityProfile(profile).to_np_array() for profile in names.values()])
    labels = list(names.keys())


    # Perform PCA and plot for each scenario
    perform_pca_and_plot(
        data,
        dimension_names,
        labels,
        None,
        2,
        "Full Vector",
        "full_vector",
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
