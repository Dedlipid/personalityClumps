import numpy as np
from personality_profile import PersonalityProfile
from raw_data import names
from general_data import dimension_names, no_overall_indices, data, labels
from pca_tools import perform_pca_and_plot

def do_pca():
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
        no_overall_indices,
        5,
        "Excluding Overall",
        "no_overall",
        3
    )

def main():
    # Perform PCA and plot for each scenario
    do_pca()
    


if __name__ == "__main__":
    main()
