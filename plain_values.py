from general_data import dark_dimension_names, dark_data, dark_labels
import plotly.express as px
import numpy as np


def plot_top_variance_components(data, labels, dimension_names, title, filename, dim=3):
    """Plots the first 'dim' components of each point in data.
    Gives 3d interactive plot and a snapshot of the plot is saved as a png file.
    """

    if dim not in [2, 3]:
        raise ValueError("dim must be 2 or 3")
    
        

    # Prepare axis labels
    axis_labels = {
        f"{axis}": name for axis, name in zip(["x", "y", "z"], dimension_names)
    }

    fig = px.scatter_3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        text=labels,
        color=labels,
        title=title,
        labels=axis_labels,
    )

    # Save the figure as an HTML file
    fig.write_html(filename)
    fig.show()


def print_covariance_matrix(data, dimension_names):
    """Prints the covariance matrix of the data."""
    covariance_matrix = np.cov(data.T)
    print("Covariance matrix:")
    print("  " + "  ".join(dimension_names))
    for i, row in enumerate(covariance_matrix):
        print(dimension_names[i], end=" ")
        for value in row:
            print(f"{value:.2f}", end=" ")
        print("")


def print_correlation_matrix(data, dimension_names):
    """Prints the correlation matrix of the data."""
    correlation_matrix = np.corrcoef(data.T)
    print("Correlation matrix:")
    print("  " + "  ".join(dimension_names))
    for i, row in enumerate(correlation_matrix):
        print(dimension_names[i], end=" ")
        for value in row:
            print(f"{value:.2f}", end=" ")
        print("")
    

def main():
    # Ensure the filename has the correct extension
    output_filename = "dark_triad.html"
    # Plot the top 3 variance components
    plot_top_variance_components(
        dark_data,
        dark_labels,
        dark_dimension_names[:3],
        "Dark Triad Factors",
        output_filename,
        dim=3,
    )
    # Print the covariance matrix
    print_correlation_matrix(dark_data, dark_dimension_names)


if __name__ == "__main__":
    main()
