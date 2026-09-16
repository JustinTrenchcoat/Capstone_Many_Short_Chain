import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f


def plot_transformed_rhat_with_f(df, distribution_name, K, M, warmups, bins=30):

    if len(warmups) != 9:
        raise ValueError(
            "warmups must contain exactly 9 Warmup Length values."
        )

    df_num = K - 1
    df_den = K * (M - 1)

    f_q05 = f.ppf(
        0.05,
        df_num,
        df_den
    )

    f_q95 = f.ppf(
        0.95,
        df_num,
        df_den
    )

    f_q999 = f.ppf(
        0.999,
        df_num,
        df_den
    )

    # --------------------------------------------------
    # Create 3 x 3 subplot grid
    # --------------------------------------------------
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 14),
        dpi=150,
        sharex=False,
        sharey=False
    )

    axes = axes.flatten()

    # --------------------------------------------------
    # Loop through warmup lengths
    # --------------------------------------------------
    for ax, warmup in zip(axes, warmups):

        # ----------------------------------------------
        # Select Rhat values
        # ----------------------------------------------
        rhat_data = df.loc[
            df["Warmup Length"] == warmup,
            "Rhat"
        ].dropna()

        # ----------------------------------------------
        # Convert to NumPy floats
        #
        # Does NOT modify the DataFrame.
        # ----------------------------------------------
        rhat_values = np.asarray(
            [
                float(x)
                for x in rhat_data
            ],
            dtype=float
        )

        # ----------------------------------------------
        # Handle missing data
        # ----------------------------------------------
        if len(rhat_values) == 0:

            ax.set_title(
                f"Warmup Length = {warmup}"
            )

            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

            continue

        # ----------------------------------------------
        # Transform Rhat into F statistic
        #
        # F = (Rhat^2 - 1) * M
        # ----------------------------------------------
        f_values = (
            (rhat_values ** 2 - 1)
            * M
        )

        # ----------------------------------------------
        # Remove invalid values
        #
        # The theoretical F distribution is defined
        # for F >= 0.
        # ----------------------------------------------
        f_values = f_values[
            np.isfinite(f_values) &
            (f_values >= 0)
        ]

        # ----------------------------------------------
        # Determine individual x-axis range
        # ----------------------------------------------
        x_min = 0.0

        x_max = max(
            f_values.max(),
            f_q999
        )

        # Add 2% padding
        if x_max > x_min:
            x_max += 0.02 * (
                x_max - x_min
            )

        # ----------------------------------------------
        # Generate theoretical F PDF
        # ----------------------------------------------
        x = np.linspace(
            x_min,
            x_max,
            2000
        )

        y = f.pdf(
            x,
            df_num,
            df_den
        )

        # ----------------------------------------------
        # Empirical histogram
        # ----------------------------------------------
        ax.hist(
            f_values,
            bins=bins,
            density=True,
            alpha=0.6,
            edgecolor="black",
            label="Transformed Rhat"
        )

        # ----------------------------------------------
        # Theoretical F PDF
        # ----------------------------------------------
        ax.plot(
            x,
            y,
            linewidth=2.5,
            label=(
                f"F({df_num}, {df_den}) PDF"
            )
        )

        # ----------------------------------------------
        # 5th percentile
        # ----------------------------------------------
        ax.axvline(
            f_q05,
            linestyle="--",
            linewidth=1.5,
            label=f"5th = {f_q05:.3f}",
            c = "red"
        )

        # ----------------------------------------------
        # 95th percentile
        # ----------------------------------------------
        ax.axvline(
            f_q95,
            linestyle="--",
            linewidth=1.5,
            label=f"95th = {f_q95:.3f}",
            c = "red"
        )

        # ----------------------------------------------
        # Labels
        # ----------------------------------------------
        ax.set_xlabel(
            r"$(\hat{R}^2 - 1)M$"
        )

        ax.set_ylabel(
            "Density"
        )

        ax.set_title(
            f"Warmup Length = {warmup}"
        )

        ax.grid(alpha=0.3)

    # --------------------------------------------------
    # Overall title
    # --------------------------------------------------
    fig.suptitle(
        f"Transformed Rhat and Theoretical F Distribution\n"
        f"Target Distribution: {distribution_name}",
        fontsize=18
    )

    # --------------------------------------------------
    # Shared legend
    # --------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    plt.tight_layout(
        rect=[0, 0, 1, 0.88]
    )

    plt.show()



def single_plot_rhat_histogram_with_pdf(
    df,
    distribution_name,
    K,
    M,
    warmup,
    bins=30
):
    """
    Plot the empirical Rhat density histogram and theoretical Rhat PDF
    for one specified Warmup Length.

    The theoretical distribution is:

        Rhat = sqrt(1 + F/M)

    where:

        F ~ F(K-1, K(M-1))

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least:
        "Rhat" and "Warmup Length"

    distribution_name : str
        Name of the target distribution.

    K : int
        Number of subchains.

    M : int
        Number of chains per subchain.

    warmup : int or float
        The Warmup Length to plot.

    bins : int
        Number of histogram bins.
    """

    # --------------------------------------------------
    # F distribution degrees of freedom
    # --------------------------------------------------
    df_num = K - 1
    df_den = K * (M - 1)

    # --------------------------------------------------
    # Select Rhat values for specified Warmup Length
    # --------------------------------------------------
    rhat_data = df.loc[
        df["Warmup Length"] == warmup,
        "Rhat"
    ].dropna()

    # --------------------------------------------------
    # Convert JAX objects to NumPy floats
    #
    # Does NOT modify the original DataFrame.
    # --------------------------------------------------
    rhat_values = np.asarray(
        [
            float(x)
            for x in rhat_data
        ],
        dtype=float
    )

    # --------------------------------------------------
    # Check data exists
    # --------------------------------------------------
    if len(rhat_values) == 0:
        raise ValueError(
            f"No Rhat values found for Warmup Length = {warmup}"
        )

    # --------------------------------------------------
    # Theoretical Rhat PDF
    #
    # Rhat = sqrt(1 + F/M)
    #
    # F = M * (Rhat^2 - 1)
    #
    # Jacobian = 2 * M * Rhat
    # --------------------------------------------------
    def rhat_pdf(r):

        r = np.asarray(r, dtype=float)

        f_value = M * (r**2 - 1)

        pdf = np.zeros_like(
            r,
            dtype=float
        )

        valid = r >= 1

        pdf[valid] = (
            f.pdf(
                f_value[valid],
                df_num,
                df_den
            )
            * 2 * M * r[valid]
        )

        return pdf

    # --------------------------------------------------
    # Calculate theoretical percentiles
    # --------------------------------------------------
    f_q05 = f.ppf(
        0.05,
        df_num,
        df_den
    )

    f_q95 = f.ppf(
        0.95,
        df_num,
        df_den
    )

    rhat_q05 = np.sqrt(
        1 + f_q05 / M
    )

    rhat_q95 = np.sqrt(
        1 + f_q95 / M
    )

    # --------------------------------------------------
    # Determine x-axis range
    #
    # Include empirical values and theoretical
    # 99.9th percentile.
    # --------------------------------------------------
    f_q999 = f.ppf(
        0.999,
        df_num,
        df_den
    )

    rhat_q999 = np.sqrt(
        1 + f_q999 / M
    )

    x_min = 1.0

    x_max = max(
        rhat_values.max(),
        rhat_q999
    )

    # Add small padding
    x_max = x_max + 0.02 * (
        x_max - x_min
    )

    # --------------------------------------------------
    # Generate theoretical PDF
    # --------------------------------------------------
    x = np.linspace(
        x_min,
        x_max,
        2000
    )

    y = rhat_pdf(x)

    # --------------------------------------------------
    # Create plot
    # --------------------------------------------------
    plt.figure(
        figsize=(10, 6),
        dpi=150
    )

    # --------------------------------------------------
    # Empirical histogram
    # --------------------------------------------------
    plt.hist(
        rhat_values,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="Empirical Rhat"
    )

    # --------------------------------------------------
    # Theoretical PDF
    # --------------------------------------------------
    plt.plot(
        x,
        y,
        linewidth=2.5,
        label="Theoretical PDF"
    )

    # --------------------------------------------------
    # 5th percentile
    # --------------------------------------------------
    plt.axvline(
        rhat_q05,
        linestyle="--",
        linewidth=2,
        label=f"5th percentile = {rhat_q05:.4f}"
    )

    # --------------------------------------------------
    # 95th percentile
    # --------------------------------------------------
    plt.axvline(
        rhat_q95,
        linestyle="--",
        linewidth=2,
        label=f"95th percentile = {rhat_q95:.4f}"
    )

    # --------------------------------------------------
    # Labels and title
    # --------------------------------------------------
    plt.xlabel("Rhat")
    plt.ylabel("Density")

    plt.title(
        f"Empirical and Theoretical Rhat Distribution\n"
        f"Target Distribution: {distribution_name} | "
        f"Warmup Length = {warmup}"
    )

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()