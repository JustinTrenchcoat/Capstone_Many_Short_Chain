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
    Warmup,
    HighlightIndex,
    bound,
    threshold,
    num_chains_short
):
    if len(Warmup) != 9:
        raise ValueError(
            "Warmup must contain exactly 9 values."
        )

    # --------------------------------------------------
    # Create 3 x 3 subplot grid
    # --------------------------------------------------

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(16, 12)
    )

    axes = axes.flatten()

    # --------------------------------------------------
    # Plot each warmup
    # --------------------------------------------------

    for i, warmup in enumerate(Warmup):

        ax = axes[i]

        # Filter by selected warmup length
        df_warmup = df[
            df["Warmup Length"] == warmup
        ].copy()

        if df_warmup.empty:
            raise ValueError(
                f"No data found for Warmup Length = {warmup}."
            )

        # --------------------------------------------------
        # Separate highlighted and unhighlighted dimensions
        # --------------------------------------------------

        highlighted = df_warmup[
            df_warmup["Dimension"].isin(HighlightIndex)
        ]

        regular = df_warmup[
            ~df_warmup["Dimension"].isin(HighlightIndex)
        ]

        # --------------------------------------------------
        # Average Rhat
        # --------------------------------------------------

        avg_rhat = df_warmup["Rhat"].mean()

        # --------------------------------------------------
        # Calculate proportion beyond threshold
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Scatter: unhighlighted dimensions
        # --------------------------------------------------

        ax.scatter(
            regular["Rhat"] - 1,
            regular["MSE"],
            alpha=0.35,
            s=25
        )

        # --------------------------------------------------
        # Scatter: highlighted dimensions
        # --------------------------------------------------

        ax.scatter(
            highlighted["Rhat"] - 1,
            highlighted["MSE"],
            alpha=0.9,
            s=40
        )

        # --------------------------------------------------
        # Horizontal reference lines
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Vertical threshold
        # --------------------------------------------------

        ax.axvline(
            threshold,
            color="blue",
            linestyle="--"
        )

        # --------------------------------------------------
        # Axis labels
        # --------------------------------------------------

        ax.set_xlabel(r"$\hat R_{\nu} - 1$")
        ax.set_ylabel("MSE")

        # Log scales
        ax.set_xscale("log")
        ax.set_yscale("log")

        # --------------------------------------------------
        # Subplot title
        # --------------------------------------------------

        ax.set_title(
            f"W={warmup}, "
            f"Avg Rhat: {avg_rhat:.4f}\n"
            f"Highlighted: {highlighted_ratio:.1%}, "
            f"Unhighlighted: {regular_ratio:.1%}"
        )

        ax.grid(alpha=0.2)

    # --------------------------------------------------
    # Overall title
    # --------------------------------------------------

    fig.suptitle(
        rf"{name}, MSE vs. $\hat R_{{\nu}}-1$",
        fontsize=18
    )

    # --------------------------------------------------
    # Shared legend outside the plots
    # --------------------------------------------------

    regular_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor="blue",
        markeredgecolor="gray",
        alpha=0.35,
        markersize=8,
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
        markersize=8,
        label="Highlighted Dimensions"
    )

    fig.legend(
        handles=[
            regular_handle,
            highlighted_handle
        ],
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    plt.subplots_adjust(
        left=0.07,
        right=0.82,
        bottom=0.07,
        top=0.90,
        wspace=0.30,
        hspace=0.40
    )

    plt.show()