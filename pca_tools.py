from sklearn.decomposition import PCA
from plot_pca import plot_pca


def pca_analysis(
    data, dimension_names, n_components=3, indices=None, print_components=False
):
    """Performs PCA and sorts components by magnitude."""
    if indices is not None:
        data = data[:, indices]
        dimension_names = [dimension_names[i] for i in indices]

    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    sorted_components = []
    for component in pca.components_:
        component_dict = {
            dim_name: dim_value
            for dim_name, dim_value in zip(dimension_names, component)
        }
        sorted_component = dict(
            sorted(component_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        )
        sorted_components.append({k: f"{v:.2f}" for k, v in sorted_component.items()})

    if print_components:
        for i, component in enumerate(sorted_components):
            print(f"Principal Component {i+1}:")
            for dim, value in component.items():
                print(f"  {dim}: {value}")

    return transformed_data, sorted_components


def perform_pca_and_plot(
    data,
    dimension_names,
    labels,
    indices,
    n_components,
    title_suffix,
    filename_suffix,
    print_components=False,
    n_draw=3,
):
    """Helper function to perform PCA and plot the results."""
    if n_draw > n_components:
        n_draw = n_components

    transformed_data, _ = pca_analysis(
        data, dimension_names, n_components, indices, print_components=print_components
    )
    plot_pca(
        transformed_data,
        labels,
        f"{n_components}D PCA of Personality Profiles ({title_suffix})",
        f"{n_components}d_pca_{filename_suffix}",
        n_draw,
    )


