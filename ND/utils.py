import torch
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def generate_timestamp_id():
    """
    Generate a unique ID based on the current timestamp.
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")  # Format: YYYYMMDDHHMMSSffffff



def plot_integrals(integrals, filename):
    """
    Plots the integrals and saves the plot to a file.

    Parameters:
        integrals (numpy.ndarray or torch.Tensor): The integrals to plot.
        filename (str): The name of the file to save the plot.
    """
    # Ensure integrals is a NumPy array
    if isinstance(integrals, torch.Tensor):
        integrals = integrals.numpy()

    n_rep = integrals.shape[0]
    n_iter = integrals.shape[1]
    time = np.arange(n_iter).reshape(-1, 1)
    time_mat = np.tile(time, [1, n_rep])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_mat, integrals.T, c="black", alpha=0.25)
    plt.xlabel("Iterations")
    plt.ylabel("Integrals")
    plt.title("Integrals vs. Iterations")
    plt.grid(True)

    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory


def write_report(varexp, column_names, varable_names, filename):
    """
    Write the report for explained variance and save it to a file.

    Parameters:
        varexp (torch.Tensor): Variance data of shape [features, columns].
        column_names (list): List of column names corresponding to the data.
        varable_names (list): List of variable names corresponding to features.
        filename (str): Filepath to save the CSV file.
    """
    df = pd.DataFrame(varexp.numpy(), columns=column_names, index=varable_names)

    logger.info(f'Data shape for plotting explained variance distributions: {df.shape}')
    logger.info(f'Column names: {column_names}')
    logger.info(f'Variables names: {varable_names[:10]}...')

    df.to_csv(filename, index_label="Variable")

    logger.info(f'Explained variance report saved to {filename}')


def plot_variance(varexp, column_names, custom_order, filename):
    """
    Plot variance and save it to a file.

    Parameters:
        varexp (torch.Tensor): Variance data of shape [features, columns].
        column_names (list): List of column names corresponding to the data.
        custom_order (list): Order of columns for the plot.
        filename (str): Filepath to save the plot.
    """
    # Prepare the data for plotting
    data = varexp
    logger.info(f'Data shape for plotting explained variance distributions: {data.shape}')
    logger.info(f'Column names: {column_names}')
    logger.info(f'Column order: {custom_order}')
    n_features = data.shape[0]
    flattened = data.reshape(-1)
    labels = column_names * n_features

    df = pd.DataFrame({
        'value': flattened.numpy(),
        'column': labels
    })

    # Set theme
    sns.set_theme(style="ticks")

    # Create the plot
    f, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(
        data=df, x="value", y="column", 
        whis=[0, 100], width=.6, order=custom_order
    )

    # Add means using a point plot
    sns.pointplot(
        data=df, x="value", y="column",
        linestyle='none', order=custom_order,
        errorbar=None, color="black", markers="D"
    )

    # Add mean values as text annotations
    means = df.groupby('column')['value'].mean()
    for i, column in enumerate(custom_order):
        mean_value = means[column]
        ax.text(
            mean_value + 0.02, i - 0.1,
            f'{mean_value:.3f}',
            color="black", ha="left", va="center", fontsize=10
        )

    # Add grid and clean up the plot
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)

    # Save the plot to a file
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory


def plot_features(
    features,
    z_linear_space,
    mu_z,
    Y,
    Y_pred,
    predicted_Ys,
    varexp,
    result_matrix,
    column_names,
    labels,
    cov,
    cov_indices,
    filename,
):
    """
    Plot features for given genes and save the plot into a file.

    Parameters:
        genes (list): List of gene indices to plot.
        z_linear_space (torch.Tensor): Linear space for z values.
        mu_z (torch.Tensor): Encoded z values.
        Y (torch.Tensor): Observed values.
        predicted_Ys: Predictions for different conditions and models.
        varexp (torch.Tensor): Variance decomposition data.
        result_matrix (pd.DataFrame): Original result matrix for feature names.
        column_names (list): Column names for the variance decomposition.
        labels (list): Labels for conditions.
        filename (str): Path to save the plot.
    """
    pal = sns.color_palette()
    pal = pal + pal + pal
    alpha = 0.7

    # Create the plot grid
    fig, axes = plt.subplots(len(features), len(column_names) + 1, figsize=(6*len(column_names), 6*len(features)))

    for a, i in enumerate(features):
        sns.lineplot(ax=axes[a, 0], x=mu_z[:, 0], y=Y_pred[:, i], legend=False, label="Prediction")
        sns.scatterplot(ax=axes[a, 0], x=mu_z[:, 0], y=Y[:, i], legend=False, alpha=alpha, label="Observed")
        axes[a, 0].set_title("ND")
        axes[a, 0].set_ylabel(f"Feature {result_matrix.columns[i]}")
        axes[a, 0].set_xlabel("z")

        for fac_index, fac in enumerate(column_names[:-1]):
            ax = axes[a, 1+fac_index]
            conds = predicted_Ys[fac]
            for condition, label in labels[fac]['names'].items():
                sns.lineplot(
                    ax=ax,
                    x=z_linear_space[:, 0],
                    y=conds[condition][:, i],  # Select points corresponding to the condition
                    color=pal[condition],
                    label=label  # Add label for the legend
                )
            sns.scatterplot(ax=ax, x=mu_z[:, 0], y=Y[:, i], hue=cov[:, cov_indices[fac]], legend=False, palette=pal, alpha=alpha, label="Observed")
            ax.set_title(fac)
            ax.set_ylabel(f"Feature {result_matrix.columns[i]}")
            ax.set_xlabel("z")

        # Variance decomposition bar plot
        axes[a, -1].bar(column_names, [varexp[i, j] for j in range(varexp.shape[1])], color=pal)
        axes[a, -1].set_title("Variance Decomposition")
        axes[a, -1].set_ylim(0, 1)

        for ax in axes[a, :-1]:
            ax.legend(title="", loc="upper right", frameon=False)

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory
