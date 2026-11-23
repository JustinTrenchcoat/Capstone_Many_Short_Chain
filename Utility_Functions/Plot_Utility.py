from scipy.stats import f
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

############################################
# General Helpers
############################################
# F distribution stats
def f_range(num_subchains, K):

    M = num_subchains / K
    df1 = K - 1
    df2 = K * (M - 1)

    f_low = f.ppf(0.05, df1, df2)
    f_high = f.ppf(0.95, df1, df2)

    rhat_low = np.sqrt(1 + f_low / M) - 1
    rhat_high = np.sqrt(1 + f_high / M) - 1
    return rhat_low, rhat_high

############################################
# Paper Replication
############################################
# Line Plots
def MSE_vs_Warmup(W_c_df, W_n_df,title, K, M):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    common = dict(
        logx=True,logy=True,
        legend=False,ax=ax,
        ylabel="Mean Sqaured Error",
        x="Warmup Length",fontsize=14)
    ax = W_c_df.plot(
        y="Avg MSE",title=title,
        linestyle="-",color="orange",**common)
    W_c_df.plot(
        y="Best MSE",linestyle="--",
        color="orange",**common)
    W_c_df.plot(
        y="Worst MSE",linestyle="--",
        color="orange",**common)
    W_n_df.plot(
        y="Avg MSE",linestyle="-",
        color="black",**common)
    W_n_df.plot(
        y="Worst MSE",linestyle="--",
        color="black",**common)
    W_n_df.plot(
        y="Best MSE",linestyle="--",
        color="black",**common)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel(common["x"], fontsize=common["fontsize"])
    ax.set_ylabel(common["ylabel"], fontsize=common["fontsize"])
    ax.legend(handles=[
        Line2D([0], [0], color="orange", lw=2, label="Constrained"),
        Line2D([0], [0], color="black", lw=2, label="Naive")])
    fig.text(
        0.02, 0.98,
        f"K = {K}, M = {M}",
        fontsize=14,
        verticalalignment="top"
    )
    plt.show()


# Scatter Plots
def MSE_vs_Rhat(df, title, naive, bound, threshold, K, num_subchains):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    ax.scatter(
        df["Rhat"] - 1,
        df["MSE"],
        color="blue",
        s=35,
        alpha=0.4,
    )

    ax.set_yscale("log")
    if not naive:
        ax.set_xscale("log")

    ax.axhline(bound[0], color="black", linestyle="--")
    ax.axhline(bound[1], color="black", linestyle="--")
    ax.axhline(1 / num_subchains, color="black")
    ax.axvline(threshold, color="blue", linestyle="--")
    ax.tick_params(axis='both', labelsize=14)

    ax.set_xlabel(r"$\widehat{R}_{\nu}-1$",fontsize=14)
    ax.set_ylabel("Scaled squared error",fontsize=14)

    suffix = "Constrained" if not naive else "Naive"
    ax.set_title(f"{title} - {suffix}", fontsize=20, fontweight="bold")

    fig.text(
            0.02, 0.98,
            f"K = {K}, M = {int(num_subchains/K)}",
            fontsize=14,
            verticalalignment="top"
    )
    
    plt.tight_layout()
    plt.show()

############################################
# Enhanced Plotting Functions for comparing Sampling Scheme
############################################

##################
# Multi-LinePlots
##################

def MSE_vs_Warmup_Jumbo(dfs,title,
                        K,M, colors, linestyles,labels):
    """Makes Comparison Line plots
    dfs, colors, linestyles, and labels are all list form for users to specify

    Example:
    dfs = [
        tfp_c_df, tfp_n_df,
        bjx_c_df, bjx_n_df,
        pf_c_df, pf_n_df]

    colors = [
        "orange", "orange",
        "red", "red",
        "blue", "blue"
        ]

    linestyles = [
        "-", "--",
        "-", "--",
        "-", "--"
    ]

    labels = [
        "TFP - Constrained",
        "TFP - Naive",
        "BlackJAX - Constrained",
        "BlackJAX - Naive",
        "PathFinder - Constrained",
        "PathFinder - Naive"
    ]

    """
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)


    def easyplot(df, color, linestyle):
    # Average MSE plots
        ax.plot(
            df["Warmup Length"], df["Avg MSE"],
            color=color, linestyle=linestyle, linewidth=2)

    for df, color, linestyle in zip(dfs, colors, linestyles):
        easyplot(df, color, linestyle)
    
    ax.set_title(title, fontsize=18, fontweight="bold", pad = 16)
    ax.set_ylabel("Mean Squared Error",fontsize=14)
    ax.set_xlabel("Warmup Length (log scale)",fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(
        which = "major",
        linestyle = "--",
        linewidth = 0.6,
        alpha = 0.4
    )

    # better aesthetic changes:
    legend_handles = [
        Line2D([0], [0], color=color, lw=2, linestyle=linestyle, label=label)
        for color, linestyle, label in zip(colors, linestyles, labels)
    ]

    ax.legend(
        handles=legend_handles,
        fontsize=12,
        frameon=True,
        loc="best")

    fig.text(
        0.02, 0.98,
        f"K = {K}, M = {M}",
        fontsize=13,
        ha = "left",
        va="top"
    )
    plt.show()
    
##################
# Multi-Scatterplots
##################

def MSE_vs_Rhat_color(dfs, titles, supertitle, bound, 
                      threshold,num_subchains, K,M,ncols):

    # ===============
    # setup
    # ===============
    rhat_low, rhat_high = f_range(num_subchains, K)
    n = len(dfs)
    nrows=int(np.ceil(n/ncols))

    # ===============
    # subplot axes
    # ===============
    fig, axes = plt.subplots(nrows,ncols, 
                             figsize=(8*ncols,5*nrows), squeeze=False,
                             dpi=300, sharex=True, sharey = True)
    axes = axes.flatten()

    # ===============
    # colormap
    # ===============
    warmups = np.sort(
        np.unique(np.concatenate(
            [df["Warmup Length"].unique() for df in dfs])))
    cmap = plt.get_cmap("coolwarm")
    color_map = {
        w:cmap(x)
        for w, x in zip(warmups, np.linspace(0,1,len(warmups)))} 

    # ===============
    # legend
    # ===============    
    handles = [
        Line2D([0],[0],marker="o", color = color_map[warmup],
               linestyle = "", markersize = 7, label=str(warmup)) for warmup in warmups]
    
    fig.legend(
        handles = handles,
        title = "Warmup Length",
        loc = "center left",
        bbox_to_anchor = (0.84, 0.5),
        fontsize = 9,
        title_fontsize = 10,
        ncol=1,
        frameon=True)
    
    # ===============
    # scatterplots
    # ===============
    for ax, df, panel_title in zip(axes, dfs, titles):
        groups = df.groupby("Warmup Length")

        for warmup, subset in groups:
            ax.scatter(
                subset["Rhat"] -1,
                subset["MSE"],
                color = color_map[warmup],
                s=28,
                alpha=0.55)
            ax.set_title(panel_title, fontsize=15, pad=8)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)    

    # ===============
    # aesthetics and reference lines
    # ===============
    for ax in axes[:n]:
        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.axhline(
            bound[0], color="grey",linestyle="--", linewidth = 1.0, alpha=0.7)
        ax.axhline(
            bound[1], color="grey",linestyle="--", linewidth = 1.0, alpha=0.7)
        ax.axhline(
            1 / num_subchains,color="black", alpha=0.7)
        ax.axvline(
            threshold, color="black",linestyle="--", linewidth = 1.2, alpha=0.8)
        ax.axvline(
            rhat_low, color = "purple", linestyle=":", linewidth = 1.2, alpha=0.8)
        ax.axvline(
            rhat_high, color = "purple", linestyle=":", linewidth = 1.2, alpha=0.8)
  
        ax.tick_params(
            axis="both",which = "both",
            labelbottom = True,labelleft = True,
            labelsize=14)
        ax.grid(
            which = "major",
            linestyle = "--",
            linewidth = 0.5,
            alpha=0.3
        )

    # ===============
    # layout
    # ===============
    fig.subplots_adjust(top=0.88,bottom=0.08,left=0.07,
                        right=0.82,hspace=0.32,wspace=0.20)
    fig.suptitle(supertitle, fontsize=20, fontweight="bold")
    fig.supxlabel(r"$\widehat{R}-1$", fontsize=16, y=0.01)
    fig.supylabel("Scaled Squared Error", fontsize=16, x=0.01)
    fig.text(
            0.02, 0.98,
            f"K = {K}, M = {int(M)}",
            fontsize=13,
            ha = "left",
            va="top"
        )
    plt.show()


####################################
# Diagnostic visualizations
####################################

##################
# Multi-LinePlots
##################

def plot_iterations(df1, df2, metric="Rhat", threshold=None,
                    bound=None, df1_name="DataFrame 1", df2_name="DataFrame 2",
                    title=None, title_prefix=None, show_df="both",
                    simplify=False, random_state=0):
    """
    Plot a selected metric across warmup lengths for 10 iterations.

    Color distinguishes dimensions; line style distinguishes DataFrames.

    show_df accepts "df1", "df2", or "both".

    If simplify=True, randomly selects 10 dimensions to plot.
    """

    valid_metrics = ["Rhat", "MSE", "B_nu", "W_nu"]

    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}")

    if metric == "MSE" and bound is not None and len(bound) != 2:
        raise ValueError("bound must contain exactly two values.")

    all_dfs = {
        "df1": (df1_name, df1, "-"),
        "df2": (df2_name, df2, "--")
    }

    if show_df not in ["df1", "df2", "both"]:
        raise ValueError("show_df must be 'df1', 'df2', or 'both'")

    selected_dfs = (
        ["df1", "df2"] if show_df == "both" else [show_df]
    )

    dfs = {
        all_dfs[key][0]: (all_dfs[key][1], all_dfs[key][2])
        for key in selected_dfs
    }

    dimensions = sorted(
        set(df1["Dimension"].unique()) |
        set(df2["Dimension"].unique())
    )

    # Randomly select 10 dimensions if simplify=True
    if simplify:
        rng = np.random.default_rng(random_state)
        dimensions = sorted(
            rng.choice(
                dimensions,
                size=min(10, len(dimensions)),
                replace=False
            )
        )

    colors = plt.cm.tab20(np.linspace(0, 1, len(dimensions)))
    dimension_colors = dict(zip(dimensions, colors))

    fig, axes = plt.subplots(
        2, 5, figsize=(22, 10), dpi=300, sharex=False
    )
    axes = axes.flatten()

    for iteration in range(10):
        ax = axes[iteration]

        for df_name, (data, linestyle) in dfs.items():
            subset = data[data["Iteration"] == iteration]

            for dimension in dimensions:
                dim_data = subset[
                    subset["Dimension"] == dimension
                ].sort_values("Warmup Length")

                if dim_data.empty:
                    continue

                ax.plot(
                    dim_data["Warmup Length"],
                    dim_data[metric],
                    color=dimension_colors[dimension],
                    linestyle=linestyle,
                    marker="o",
                    markersize=3,
                    label=f"{df_name}, Dim {dimension}"
                )

        # Reference lines
        if metric == "Rhat" and threshold is not None:
            ax.axhline(
                y=threshold,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label="Threshold"
            )

        elif metric == "MSE" and bound is not None:
            lowerbound, upperbound = bound

            ax.axhline(
                y=lowerbound,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label="Lower bound"
            )

            ax.axhline(
                y=upperbound,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label="Upper bound"
            )

        ax.set_title(f"Iteration {iteration}")
        ax.set_xlabel("Warmup Length")
        ax.set_ylabel(metric)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9
    )

    # Construct supertitle
    main_title = title if title is not None else f"{metric} vs. Warmup Length"

    title_suffix = (
        "Comparison" if show_df == "both"
        else df1_name if show_df == "df1"
        else df2_name
    )

    full_title = " ".join(
        part for part in [title_prefix, main_title, title_suffix]
        if part
    )

    fig.suptitle(full_title, fontsize=16, y=1.12)

    plt.tight_layout()
    plt.show()

##################
# Single Iteration
##################

def plot_single_iteration(
    df1, df2,
    iteration=0,
    metric="Rhat",
    threshold=None,
    bound=None,
    df1_name="DataFrame 1",
    df2_name="DataFrame 2",
    title=None,
    title_prefix=None,
    show_df="both",
    simplify=False,
    random_state=0
):
    """
    Plot a selected metric across warmup lengths for one iteration.

    Color distinguishes dimensions; line style distinguishes DataFrames.

    Parameters
    ----------
    iteration : int
        The iteration to plot.
    show_df : str
        'df1', 'df2', or 'both'.
    simplify : bool
        If True, randomly select 10 dimensions to plot.
    random_state : int
        Random seed for selecting dimensions when simplify=True.
    """

    valid_metrics = ["Rhat", "MSE", "B_nu", "W_nu"]

    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}")

    if metric == "MSE" and bound is not None and len(bound) != 2:
        raise ValueError("bound must contain exactly two values.")

    all_dfs = {
        "df1": (df1_name, df1, "-"),
        "df2": (df2_name, df2, "--")
    }

    if show_df not in ["df1", "df2", "both"]:
        raise ValueError("show_df must be 'df1', 'df2', or 'both'")

    selected_dfs = (
        ["df1", "df2"] if show_df == "both" else [show_df]
    )

    dfs = {
        all_dfs[key][0]: (all_dfs[key][1], all_dfs[key][2])
        for key in selected_dfs
    }

    dimensions = sorted(
        set(df1["Dimension"].unique()) |
        set(df2["Dimension"].unique())
    )

    # Randomly select 10 dimensions if simplify=True
    if simplify:
        rng = np.random.default_rng(random_state)
        dimensions = sorted(
            rng.choice(
                dimensions,
                size=min(10, len(dimensions)),
                replace=False
            )
        )

    colors = plt.cm.tab20(np.linspace(0, 1, len(dimensions)))
    dimension_colors = dict(zip(dimensions, colors))

    # Single subplot
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    for df_name, (data, linestyle) in dfs.items():
        subset = data[data["Iteration"] == iteration]

        for dimension in dimensions:
            dim_data = subset[
                subset["Dimension"] == dimension
            ].sort_values("Warmup Length")

            if dim_data.empty:
                continue

            ax.plot(
                dim_data["Warmup Length"],
                dim_data[metric],
                color=dimension_colors[dimension],
                linestyle=linestyle,
                marker="o",
                markersize=3,
                label=f"{df_name}, Dim {dimension}"
            )

    # Reference lines
    if metric == "Rhat" and threshold is not None:
        ax.axhline(
            y=threshold,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="Threshold"
        )

    elif metric == "MSE" and bound is not None:
        lowerbound, upperbound = bound

        ax.axhline(
            y=lowerbound,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="Lower bound"
        )

        ax.axhline(
            y=upperbound,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="Upper bound"
        )

    ax.set_title(f"Iteration {iteration}")
    ax.set_xlabel("Warmup Length")
    ax.set_ylabel(metric)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(
        loc="best",
        fontsize=10,
        frameon=True
    )

    # Construct supertitle
    main_title = title if title is not None else f"{metric} vs. Warmup Length"

    title_suffix = (
        "Comparison" if show_df == "both"
        else df1_name if show_df == "df1"
        else df2_name
    )

    full_title = " ".join(
        part for part in [title_prefix, main_title, title_suffix]
        if part
    )

    fig.suptitle(full_title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()

####################################
# Enhanced Plotting Functions for comparing target distributions
####################################


##################
# Helper
##################
def loglog_correlation(df):
    """
    Pearson correlation between log(Rhat - 1) and log(MSE).
    """
    x = np.asarray(df["Rhat"], dtype=np.float64) - 1
    y = np.asarray(df["MSE"], dtype=np.float64)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )

    log_x = np.log(x[mask])
    log_y = np.log(y[mask])

    if len(log_x) < 2:
        return np.nan

    if np.std(log_x) == 0 or np.std(log_y) == 0:
        return np.nan

    return np.corrcoef(log_x, log_y)[0, 1]

##################
# Multi-Scatterplots
##################

def MSE_vs_Rhat_8panel(dfs, titles,bound, threshold, num_subchains, K, supertitle):
    # ==============
    # Validate inputs
    # ==============
    if len(dfs) != 8:
        raise ValueError("dfs must contain exactly 8 dataframes.")

    rhat_low, rhat_high = f_range(num_subchains, K)
    M = num_subchains/K
    # ==============
    # Figure setup
    # ==============
    fig, axes = plt.subplots(
        2, 4,
        figsize=(28, 12),
        dpi=300,
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    # ==============
    # Warmup color mapping
    # ==============
    warmups = np.sort(
        np.unique(
            np.concatenate([df["Warmup Length"].unique() for df in dfs])
        )
    )

    cmap = plt.get_cmap("coolwarm")
    color_map = {
        w: cmap(x)
        for w, x in zip(
            warmups,
            np.linspace(0, 1, len(warmups))
        )
    }

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color=color_map[warmup],
            linestyle="",
            markersize=7,
            label=str(warmup)
        )
        for warmup in warmups
    ]

    # ==============
    # Scatterplots
    # ==============
    for ax, df, panel_title in zip(axes, dfs, titles):
      groups = df.groupby("Warmup Length")

      for warmup, subset in groups:
          ax.scatter(
              subset["Rhat"] - 1,
              subset["MSE"],
              color=color_map[warmup],
              s=28,
              alpha=0.55
          )

      corr = loglog_correlation(df)

      corr_text = (
          f"r = {corr:.3f}"
          if np.isfinite(corr)
          else "r = N/A"
      )

      ax.set_title(
          f"{panel_title}\n{corr_text}",
          fontsize=15,
          pad=8
      )

    # ==============
    # Warmup legend
    # ==============
    fig.legend(
        handles=handles,
        title="Warmup Length",
        loc="center left",
        bbox_to_anchor=(0.90, 0.5),
        fontsize=10,
        title_fontsize=11,
        ncol=1,
        frameon=True
    )

    # ==============
    # Reference lines and axes
    # ==============
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xscale("log")

        # MSE bounds
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

        # 1 / number of subchains
        ax.axhline(
            1 / num_subchains,
            color="black",
            linewidth=1.0,
            alpha=0.7
        )

        # Empirical R-hat threshold
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8
        )

        # F-derived R-hat boundaries
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

        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=True,
            labelleft=True,
            labelsize=12
        )

        ax.grid(
            which="major",
            linestyle="--",
            linewidth=0.5,
            alpha=0.3
        )

    # ==============
    # Figure layout
    # ==============
    fig.subplots_adjust(
        top=0.88,
        bottom=0.10,
        left=0.07,
        right=0.87,
        hspace=0.32,
        wspace=0.20
    )

    fig.suptitle(
        supertitle,
        fontsize=20,
        fontweight="bold"
    )

    fig.supxlabel(
        r"$\widehat{R}-1$",
        fontsize=16,
        y=0.02
    )
    fig.supylabel(
        "Scaled Squared Error",
        fontsize=16,
        x=0.02
    )

    fig.text(
        0.02, 0.98,
        f"K = {K}, M = {int(M)}",
        fontsize=13,
        ha="left",
        va="top"
    )
    fig.suptitle(
    rf"{supertitle}, MSE vs. $\widehat{{R}}_{{\nu}}$",
    fontsize=20,
    fontweight="bold"
    )

    plt.show()

####################################
# Enhanced Plotting Functions for comparing across warmup lengths
####################################

##################
# Multi-Scatterplots
##################

def plot_mse_vs_rhat_9(
    name, df, warmup, bound, threshold,
    num_chains_short, num_super_chains,
    color_by=False
):

    if len(warmup) != 9:
        raise ValueError("warmup must contain exactly 9 values.")

    if color_by not in (False, "Dimension", "Iteration"):
        raise ValueError(
            "color_by must be False, 'Dimension', or 'Iteration'."
        )

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
        dpi=300,
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    # ==============================================
    # Optional continuous color scale
    # ==============================================

    if color_by is not False:

        if color_by == "Dimension":
            color_column = "Dimension"
        else:
            color_column = "Iteration"

        if color_column not in df.columns:
            raise ValueError(
                f"Column '{color_column}' not found in df."
            )

        color_values = pd.to_numeric(
            df[color_column], errors="coerce"
        )

        if color_values.notna().sum() == 0:
            raise ValueError(
                f"Column '{color_column}' contains no valid numeric values."
            )

        color_min = color_values.min()
        color_max = color_values.max()

        if color_min == color_max:
            norm = plt.Normalize(
                vmin=color_min - 0.5,
                vmax=color_max + 0.5
            )
        else:
            norm = plt.Normalize(
                vmin=color_min,
                vmax=color_max
            )

        cmap = plt.get_cmap("turbo")

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
        # Average Rhat
        # ------------------------------------------

        avg_rhat = df_warmup["Rhat"].mean()

        # ------------------------------------------
        # Scatterplot
        # ------------------------------------------

        if color_by is False:

            ax.scatter(
                df_warmup["Rhat"] - 1,
                df_warmup["MSE"],
                color="blue",
                alpha=0.35,
                s=25
            )

        else:

            ax.scatter(
                df_warmup["Rhat"] - 1,
                df_warmup["MSE"],
                c=pd.to_numeric(
                    df_warmup[color_column],
                    errors="coerce"
                ),
                cmap=cmap,
                norm=norm,
                alpha=0.65,
                s=25
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

        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8
        )

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
            f"W={w}, Avg Rhat: {avg_rhat:.4f}",
            fontsize=10,
            pad=8
        )

    # ==============================================
    # Figure layout
    # ==============================================

    fig.subplots_adjust(
        top=0.88,
        bottom=0.10,
        left=0.07,
        right=0.88 if color_by is not False else 0.95,
        hspace=0.36,
        wspace=0.20
    )

    # ==============================================
    # Optional shared colorbar
    # ==============================================

    if color_by is not False:

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )
        sm.set_array([])

        cbar_ax = fig.add_axes([0.90, 0.20, 0.015, 0.60])

        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(color_by, fontsize=13)
        cbar.ax.tick_params(labelsize=10)

    # ==============================================
    # Figure labels
    # ==============================================

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

##################
# Single Highlight Plot
##################

def draw_highlight_subplot(
    ax, df, w, HighlightIdx1, HighlightIdx2,
    bound, threshold, num_chains_short,
    num_super_chains, show_legend=False
):
    """
    Draw one warmup-specific subplot, including F-derived
    Rhat boundaries. Does not create a figure or legend.
    """
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
    # Filter data for the selected warmup
    # ==============================================
    df_warmup = df[df["Warmup Length"] == w].copy()

    if df_warmup.empty:
        raise ValueError(f"No data found for Warmup Length = {w}.")

    # ==============================================
    # Separate groups
    # ==============================================
    idx1 = set(HighlightIdx1)
    idx2 = set(HighlightIdx2) - idx1

    highest = df_warmup[df_warmup["Dimension"].isin(idx1)]
    medium = df_warmup[df_warmup["Dimension"].isin(idx2)]
    low = df_warmup[
        ~df_warmup["Dimension"].isin(idx1 | idx2)
    ]

    # ==============================================
    # Average Rhat
    # ==============================================
    avg_rhat = df_warmup["Rhat"].mean()

    # ==============================================
    # Average percentage exceeding threshold
    # ==============================================
    def avg_exceedance_percentage_per_iteration(group):
        if group.empty:
            return 0.0

        exceeds = ((group["Rhat"] - 1) > threshold).astype(int)
        percentages = exceeds.groupby(group["Iteration"]).mean() * 100

        return percentages.mean()

    high_pct = avg_exceedance_percentage_per_iteration(highest)
    med_pct = avg_exceedance_percentage_per_iteration(medium)
    low_pct = avg_exceedance_percentage_per_iteration(low)

    # ==============================================
    # Scatterplots
    # ==============================================
    ax.scatter(
        low["Rhat"] - 1, low["MSE"],
        color="blue", alpha=0.30, s=25,
        label="Low variance"
    )

    ax.scatter(
        medium["Rhat"] - 1, medium["MSE"],
        color="orange", alpha=0.45, s=35,
        label="Medium variance"
    )

    ax.scatter(
        highest["Rhat"] - 1, highest["MSE"],
        color="red", alpha=0.50, s=40,
        label="Highest variance"
    )

    # ==============================================
    # Reference lines: MSE
    # ==============================================
    for i, y in enumerate(bound):
      ax.axhline(
          y,
          color="grey",
          linestyle="--",
          linewidth=1.0,
          alpha=0.7,
          label="MSE bounds" if i == 0 else "_nolegend_"
      )

    ax.axhline(
      1 / num_chains_short,
      color="black",
      linewidth=1.0,
      alpha=0.7,
      label=r"$1/N_{\mathrm{short}}$"
    )

    # ==============================================
    # Reference lines: Rhat
    # ==============================================
    ax.axvline(
      threshold,
      color="black",
      linestyle="--",
      linewidth=1.2,
      alpha=0.8,
      label="Empirical threshold"
    )

    for i, x in enumerate((rhat_low, rhat_high)):
      ax.axvline(
          x,
          color="purple",
          linestyle=":",
          linewidth=1.2,
          alpha=0.8,
          label="F-derived boundaries" if i == 0 else "_nolegend_"
      )

    # ==============================================
    # Axes formatting
    # ==============================================
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.tick_params(
        axis="both", which="both",
        labelbottom=True, labelleft=True,
        labelsize=11
    )

    ax.grid(
        which="major", linestyle="--",
        linewidth=0.5, alpha=0.3
    )

    # ==============================================
    # Subplot title
    # ==============================================
    ax.set_title(
        f"W={w}, Avg Rhat: {avg_rhat:.4f}\n"
        f"Highest: {high_pct:.2f}%, "
        f"Medium: {med_pct:.2f}%, "
        f"Low: {low_pct:.1f}% exceeding threshold",
        fontsize=10, pad=8
    )

    if show_legend:
        ax.legend(
            loc="best",
            fontsize=8,
            frameon=True
        )
    return ax

##################
# Multi-Highlight Plots 
##################

def highlightPlots(
    name, df, warmup, HighlightIdx1, HighlightIdx2,
    bound, threshold, num_chains_short, num_super_chains
):
    if len(warmup) != 9:
        raise ValueError("warmup must contain exactly 9 values.")

    K = num_super_chains
    M = num_chains_short / num_super_chains

    if M <= 1 or K <= 1:
        raise ValueError("K and M must both be greater than 1.")

    # ==============================================
    # Figure setup
    # ==============================================
    fig, axes = plt.subplots(
        3, 3,
        figsize=(22, 15),
        dpi=300,
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    # ==============================================
    # Draw each subplot
    # ==============================================
    for ax, w in zip(axes, warmup):
        draw_highlight_subplot(
            ax=ax,
            df=df,
            w=w,
            HighlightIdx1=HighlightIdx1,
            HighlightIdx2=HighlightIdx2,
            bound=bound,
            threshold=threshold,
            num_chains_short=num_chains_short,
            num_super_chains=num_super_chains
        )

    # ==============================================
    # Shared legend
    # ==============================================
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="red",
            linestyle="", alpha=0.5, markersize=8,
            label="Highest variance"
        ),
        Line2D(
            [0], [0], marker="o", color="orange",
            linestyle="", alpha=0.45, markersize=8,
            label="Medium variance"
        ),
        Line2D(
            [0], [0], marker="o", color="blue",
            linestyle="", alpha=0.3, markersize=8,
            label="Low variance"
        ),
        Line2D(
            [0], [0], color="black", linestyle="--",
            linewidth=1.2, label="Empirical threshold"
        ),
        Line2D(
            [0], [0], color="purple", linestyle=":",
            linewidth=1.2, label="F-derived boundaries"
        ),
        Line2D(
            [0], [0], color="grey", linestyle="--",
            linewidth=1.0, label="MSE bounds"
        ),
        Line2D(
            [0], [0], color="black", linestyle="-",
            linewidth=1.0, label=r"$1/N_{\mathrm{short}}$"
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