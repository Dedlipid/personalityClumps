from general_data import dimension_names, no_overall_indices, data, labels
from k_cluster import perform_encoding_and_draw
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
        3,
        "Excluding Overall",
        "no_overall",
    )


def do_k_cluster_encoding():
  # Perform k-cluster encoding and draw the 3D visualization
  perform_encoding_and_draw(data, labels, k=3)

def main():
    # Perform PCA and plot for each scenario
    do_pca()
    do_k_cluster_encoding()


if __name__ == "__main__":
    main()
