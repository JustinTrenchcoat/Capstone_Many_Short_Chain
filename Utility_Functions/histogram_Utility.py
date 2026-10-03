import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

def plot_transformed_rhat_with_f(
    df,
    distribution_name,
    K,
    M,
    warmups,
    bins="auto"
):

    if len(warmups) != 9:
        raise ValueError(
            "warmups must contain exactly 9 Warmup Length values."
        )

    df_num = K - 1
    df_den = K * (M - 1)

    # --------------------------------------------------
    # Theoretical F quantiles
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
        dpi=300,
        sharex=False,
        sharey=False
    )

    axes = axes.flatten()

    # --------------------------------------------------
    # Loop through warmup lengths
    # --------------------------------------------------
    for ax, warmup in zip(axes, warmups):

        # ----------------------------------------------
        # Select rows for this warmup length
        # ----------------------------------------------
        warmup_data = df.loc[
            df["Warmup Length"] == warmup
        ]

        # ----------------------------------------------
        # Select Rhat values
        # ----------------------------------------------
        rhat_data = warmup_data[
            "Rhat"
        ].dropna()

        # ----------------------------------------------
        # Convert Rhat to NumPy floats
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
        # Calculate average MSE
        # ----------------------------------------------
        mse_data = warmup_data[
            "MSE"
        ].dropna()

        mse_values = np.asarray(
            [
                float(x)
                for x in mse_data
            ],
            dtype=float
        )

        if len(mse_values) > 0:
            avg_mse = np.mean(mse_values)
        else:
            avg_mse = np.nan

        # ----------------------------------------------
        # Handle missing Rhat data
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
        # ----------------------------------------------
        f_values = f_values[
            np.isfinite(f_values) &
            (f_values >= 0)
        ]

        # ----------------------------------------------
        # Handle case where no valid F values remain
        # ----------------------------------------------
        if len(f_values) == 0:

            ax.set_title(
                f"Warmup Length = {warmup}"
            )

            ax.text(
                0.5,
                0.5,
                "No valid F values",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

            continue

        # ==================================================
        # Empirical distribution
        # ==================================================

        # ----------------------------------------------
        # Overflow cutoff
        #
        # Everything above the theoretical 99.9th
        # percentile goes into one overflow bin.
        # ----------------------------------------------
        overflow_cutoff = f_q999

        # Main region
        f_main = f_values[
            f_values <= overflow_cutoff
        ]

        # Overflow region
        f_overflow = f_values[
            f_values > overflow_cutoff
        ]

        # ----------------------------------------------
        # Determine automatic histogram bins
        #
        # IMPORTANT:
        # The extreme values are excluded when determining
        # the automatic bin widths, so they cannot stretch
        # the histogram.
        # ----------------------------------------------
        if len(f_main) > 0:

            bin_edges = np.histogram_bin_edges(
                f_main,
                bins=bins
            )

            # Make the final regular bin terminate
            # exactly at the overflow cutoff.
            bin_edges[-1] = overflow_cutoff

        else:

            # Extremely unlikely, but handle the case
            # where every observation is an overflow value.
            bin_edges = np.array([
                0.0,
                overflow_cutoff
            ])

        # ----------------------------------------------
        # Calculate counts for normal bins
        # ----------------------------------------------
        if len(f_main) > 0:

            counts, _ = np.histogram(
                f_main,
                bins=bin_edges
            )

        else:

            counts = np.zeros(
                len(bin_edges) - 1,
                dtype=int
            )

        # ----------------------------------------------
        # Convert counts to probability density
        #
        # Integral of the histogram over all bins
        # will approximately equal 1, including overflow.
        # ----------------------------------------------
        widths = np.diff(bin_edges)

        density = (
            counts
            / len(f_values)
            / widths
        )

        # ----------------------------------------------
        # Plot normal bins
        # ----------------------------------------------
        ax.bar(
            bin_edges[:-1],
            density,
            width=widths,
            align="edge",
            alpha=0.6,
            edgecolor="black",
            label="Transformed Rhat"
        )

        # ----------------------------------------------
        # Overflow bin
        # ----------------------------------------------
        if len(f_overflow) > 0:

            # Give the overflow bin the same visual width
            # as the final regular bin.
            bin_width = widths[-1]

            overflow_density = (
                len(f_overflow)
                / len(f_values)
                / bin_width
            )

            ax.bar(
                overflow_cutoff,
                overflow_density,
                width=bin_width,
                align="edge",
                alpha=0.6,
                edgecolor="black"
            )

            # Label overflow bin
            ax.text(
                overflow_cutoff + bin_width / 2,
                overflow_density,
                f">{overflow_cutoff:.2f}\n"
                f"(n={len(f_overflow)})",
                ha="center",
                va="bottom",
                fontsize=9
            )

        # ----------------------------------------------
        # Determine x-axis range
        #
        # The overflow bin is the rightmost visible bin.
        # Extreme observations do not stretch the axis.
        # ----------------------------------------------
        bin_width = widths[-1]

        x_min = 0.0

        x_max = (
            overflow_cutoff
            + bin_width
        )

        # ==================================================
        # Theoretical F distribution
        # ==================================================

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

        ax.plot(
            x,
            y,
            linewidth=2.5,
            label=f"F({df_num}, {df_den}) PDF"
        )

        # ----------------------------------------------
        # 5th percentile
        # ----------------------------------------------
        ax.axvline(
            f_q05,
            linestyle="--",
            linewidth=1.5,
            label=f"5th = {f_q05:.3f}",
            c="red"
        )

        # ----------------------------------------------
        # 95th percentile
        # ----------------------------------------------
        ax.axvline(
            f_q95,
            linestyle="--",
            linewidth=1.5,
            label=f"95th = {f_q95:.3f}",
            c="red"
        )

        # ==================================================
        # Labels
        # ==================================================

        ax.set_xlabel(
            r"$(\hat{R}^2 - 1)M$"
        )

        ax.set_ylabel(
            "Density"
        )

        ax.set_title(
            f"Warmup Length = {warmup}"
        )

        # ----------------------------------------------
        # Display average MSE
        # ----------------------------------------------
        if np.isfinite(avg_mse):

            ax.text(
                0.97,
                0.95,
                f"Avg MSE = {avg_mse:.4g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                bbox=dict(
                    boxstyle="round",
                    alpha=0.8
                )
            )

        else:

            ax.text(
                0.97,
                0.95,
                "Avg MSE = N/A",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11
            )

        ax.grid(
            alpha=0.3
        )

        ax.set_xlim(
            x_min,
            x_max
        )

    # ==================================================
    # Overall title
    # ==================================================

    fig.suptitle(
        f"Transformed Rhat and Theoretical F Distribution\n"
        f"Target Distribution: {distribution_name}",
        fontsize=18
    )

    # ==================================================
    # Shared legend
    # ==================================================

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4
    )

    # ==================================================
    # Layout
    # ==================================================

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