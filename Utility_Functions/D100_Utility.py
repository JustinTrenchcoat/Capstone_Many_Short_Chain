import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_mse_vs_rhat(
    name,
    df,
    Warmup,
    HighlightIndex,
    bound,
    threshold,
    num_chains_short
):
    # Filter by selected warmup length
    df_warmup = df[df["Warmup Length"] == Warmup].copy()

    if df_warmup.empty:
        raise ValueError(
            f"No data found for Warmup Length = {Warmup}."
        )

    # Separate highlighted and unhighlighted dimensions
    highlighted = df_warmup[
        df_warmup["Dimension"].isin(HighlightIndex)
    ]

    regular = df_warmup[
        ~df_warmup["Dimension"].isin(HighlightIndex)
    ]

    # Average Rhat
    avg_rhat = df_warmup["Rhat"].mean()

    # Calculate proportion of points beyond threshold
    highlighted_total = len(highlighted)
    regular_total = len(regular)

    highlighted_beyond = (
        (highlighted["Rhat"] - 1) > threshold
    ).sum()

    regular_beyond = (
        (regular["Rhat"] - 1) > threshold
    ).sum()

    highlighted_ratio = (
        highlighted_beyond / highlighted_total
        if highlighted_total > 0 else 0
    )

    regular_ratio = (
        regular_beyond / regular_total
        if regular_total > 0 else 0
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter: unhighlighted dimensions
    ax.scatter(
        regular["Rhat"] - 1,
        regular["MSE"],
        alpha=0.35,
        label="Unhighlighted Dimensions"
    )

    # Scatter: highlighted dimensions
    ax.scatter(
        highlighted["Rhat"] - 1,
        highlighted["MSE"],
        alpha=0.9,
        s=50,
        label="Highlighted Dimensions"
    )

    # Horizontal reference lines
    ax.axhline(
        bound[0],
        color="black",
        linestyle="--"
    )

    ax.axhline(
        bound[1],
        color="black",
        linestyle="--"
    )

    ax.axhline(
        1 / num_chains_short,
        color="black"
    )

    # Vertical threshold
    ax.axvline(
        threshold,
        color="blue",
        linestyle="--"
    )

    # Axis labels
    ax.set_xlabel("Rhat - 1")
    ax.set_ylabel("MSE")

    # Log-scaled x-axis
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Title
    ax.set_title(
        f"{name}: MSE vs Rhat - 1\n"
        f"Warmup Length = {Warmup}, "
        f"Avg Rhat = {avg_rhat:.4f}"
    )

    # --------------------------------------------------
    # Combined legend + information box
    # --------------------------------------------------

    regular_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor="blue",
        markeredgecolor="gray",
        alpha=0.35,
        markersize=7,
        label="Unhighlighted Dimensions"
    )

    highlighted_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor="orange",
        markeredgecolor="gray",
        alpha=0.9,
        markersize=7,
        label="Highlighted Dimensions"
    )

    legend = ax.legend(
        handles=[
            regular_handle,
            highlighted_handle
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False
    )

    # Information box below the legend
    ratio_text = (
        f"Beyond threshold:\n"
        f"Highlighted:   "
        f"{highlighted_beyond}/{highlighted_total} "
        f"({highlighted_ratio:.1%})\n"
        f"Unhighlighted: "
        f"{regular_beyond}/{regular_total} "
        f"({regular_ratio:.1%})"
    )

    ax.text(
        1.02,
        0.73,
        ratio_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )

    # Leave space on the right for legend + information
    plt.subplots_adjust(right=0.72)

    ax.grid(alpha=0.2)

    plt.show()
# ================================================================================
def plot_mse_vs_rhat_9(
    name,
    df,
    warmup,
    HighlightIdx1,
    HighlightIdx2,
    bound,
    threshold,
    num_chains_short,
    num_super_chains
):
    if len(warmup) != 9:
        raise ValueError("warmup must contain exactly 9 values.")

    # ==============================================
    # K, M, and F-derived Rhat boundaries
    # ==============================================

    K = num_super_chains
    M = num_chains_short / num_super_chains

    if M <= 1 or K <= 1:
        raise ValueError("K and M must both be greater than 1.")

    df1 = K - 1
    df2 = K * (M - 1)

    f_low = f.ppf(0.05, df1, df2)
    f_high = f.ppf(0.95, df1, df2)

    rhat_low = np.sqrt(1 + f_low / M) - 1
    rhat_high = np.sqrt(1 + f_high / M) - 1

    # ==============================================
    # Figure setup
    # ==============================================

    fig, axes = plt.subplots(
        3, 3,
        figsize=(22, 15),
        dpi=150,
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    # ==============================================
    # Highlight groups
    # ==============================================

    idx1 = set(HighlightIdx1)
    idx2 = set(HighlightIdx2) - idx1

    # ==============================================
    # Plot each warmup
    # ==============================================

    for i, w in enumerate(warmup):

        ax = axes[i]

        df_warmup = df[
            df["Warmup Length"] == w
        ].copy()

        if df_warmup.empty:
            raise ValueError(
                f"No data found for Warmup Length = {w}."
            )

        # ------------------------------------------
        # Separate groups
        # ------------------------------------------

        highest = df_warmup[
            df_warmup["Dimension"].isin(idx1)
        ]

        medium = df_warmup[
            df_warmup["Dimension"].isin(idx2)
        ]

        low = df_warmup[
            ~df_warmup["Dimension"].isin(idx1 | idx2)
        ]

        # ------------------------------------------
        # Average Rhat
        # ------------------------------------------

        avg_rhat = df_warmup["Rhat"].mean()

        # ------------------------------------------
        # Average number exceeding threshold
        # per iteration, separately by group
        # ------------------------------------------

        def avg_exceedance_percentage_per_iteration(group):
          if group.empty:
            return 0.0

          group = group.copy()
          group["Exceeds"] = (
              (group["Rhat"] - 1) > threshold
          ).astype(int)

          percentages = (
              group.groupby("Iteration")["Exceeds"].mean()* 100)

          return percentages.mean()

        high_pct = avg_exceedance_percentage_per_iteration(highest)
        med_pct = avg_exceedance_percentage_per_iteration(medium)
        low_pct = avg_exceedance_percentage_per_iteration(low)

        # ------------------------------------------
        # Scatterplots
        # ------------------------------------------

        ax.scatter(
            low["Rhat"] - 1,
            low["MSE"],
            color="blue",
            alpha=0.30,
            s=25,
            label="Low variance"
        )

        ax.scatter(
            medium["Rhat"] - 1,
            medium["MSE"],
            color="orange",
            alpha=0.45,
            s=35,
            label="Medium variance"
        )

        ax.scatter(
            highest["Rhat"] - 1,
            highest["MSE"],
            color="red",
            alpha=0.50,
            s=40,
            label="Highest variance"
        )

        # ------------------------------------------
        # Reference lines: MSE bounds
        # ------------------------------------------

        ax.axhline(
            bound[0],
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7
        )

        ax.axhline(
            bound[1],
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7
        )

        # 1 / number of short chains

        ax.axhline(
            1 / num_chains_short,
            color="black",
            linewidth=1.0,
            alpha=0.7
        )

        # ------------------------------------------
        # Reference lines: Rhat
        # ------------------------------------------

        # Empirical threshold
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8
        )

        # F-derived boundaries
        ax.axvline(
            rhat_low,
            color="purple",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8
        )

        ax.axvline(
            rhat_high,
            color="purple",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8
        )

        # ------------------------------------------
        # Axes
        # ------------------------------------------

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=True,
            labelleft=True,
            labelsize=11
        )

        ax.grid(
            which="major",
            linestyle="--",
            linewidth=0.5,
            alpha=0.3
        )

        # ------------------------------------------
        # Subplot title
        # ------------------------------------------

        ax.set_title(
            f"W={w}, Avg Rhat: {avg_rhat:.4f}\n"
            f"Highest: {high_pct:.2f}%, "
            f"Medium: {med_pct:.2f}%, "
            f"Low: {low_pct:.1f}% exceeding threshold",
            fontsize=10,
            pad=8
        )

    # ==============================================
    # Shared legend
    # ==============================================

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="red",
            linestyle="",
            alpha=0.5,
            markersize=8,
            label="Highest variance"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="orange",
            linestyle="",
            alpha=0.45,
            markersize=8,
            label="Medium variance"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="blue",
            linestyle="",
            alpha=0.3,
            markersize=8,
            label="Low variance"
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Empirical threshold"
        ),
        Line2D(
            [0], [0],
            color="purple",
            linestyle=":",
            linewidth=1.2,
            label="F-derived boundaries"
        ),
        Line2D(
            [0], [0],
            color="grey",
            linestyle="--",
            linewidth=1.0,
            label="MSE bounds"
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="-",
            linewidth=1.0,
            label=r"$1/N_{\mathrm{short}}$"
        )
    ]

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.87, 0.5),
        fontsize=10,
        frameon=True
    )

    # ==============================================
    # Figure layout and labels
    # ==============================================

    fig.subplots_adjust(
        top=0.88,
        bottom=0.10,
        left=0.07,
        right=0.85,
        hspace=0.36,
        wspace=0.20
    )

    fig.suptitle(
        rf"{name}: MSE vs. $\widehat{{R}}_{{\nu}}-1$",
        fontsize=20,
        fontweight="bold"
    )

    fig.supxlabel(
        r"$\widehat{R}_{\nu}-1$",
        fontsize=16,
        y=0.025
    )

    fig.supylabel(
        "Scaled Squared Error",
        fontsize=16,
        x=0.02
    )

    fig.text(
        0.02, 0.98,
        f"K = {K}, M = {M:g}",
        fontsize=13,
        ha="left",
        va="top"
    )

    plt.show()