from scipy.stats import f
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

############################################
# Paper Replication
############################################
# Line Plots
def MSE_vs_Warmup(W_c_df, W_n_df,title, K, M):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
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
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

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
# Enhanced Plotting Functions
############################################
def MSE_vs_Warmup_Jumbo(tfp_c_df, tfp_n_df, 
                        bjx_c_df, bjx_n_df,
                        pf_c_df, pf_n_df,title,
                        K,M):
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    colors = {
           "TFP":"orange",
           "BlackJAX":"red",
           "PathFinder":"blue"}
    linestyles = {
           "Constrained":"-",
           "Naive":"--"}

    def easyplot(df, color, linestyle):
    # Average
        ax.plot( df["Warmup Length"], df["Avg MSE"],
                color=color, linestyle=linestyle, linewidth=2)

    # TFP implementation:
    easyplot(
        tfp_c_df,colors["TFP"],
        linestyles["Constrained"])
    easyplot(
        tfp_n_df,colors["TFP"],
        linestyles["Naive"])
    # BlackJAX implementation
    easyplot(
        bjx_c_df,colors["BlackJAX"],
        linestyles["Constrained"])
    easyplot(
        bjx_n_df,colors["BlackJAX"],linestyles["Naive"])
    # Pathfinder initializations
    easyplot(
        pf_c_df, colors["PathFinder"], linestyles["Constrained"])
    easyplot(
        pf_n_df,colors["PathFinder"],linestyles["Naive"])
    
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
        Line2D([0], [0], color="orange", lw=2, label="TFP"),
        Line2D([0], [0], color="red", lw=2, label="BlackJAX"),
        Line2D([0], [0], color="blue", lw=2, label="PathFinder Initialization"),
        Line2D([0], [0], color="black", lw=2,
           linestyle="-", label="Constrained"),
        Line2D([0], [0], color="black", lw=2,
           linestyle="--", label="Naive"),
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
    
# Color Coded Scatter Plots
def MSE_vs_Rhat_color(dfs, titles, supertitle, bound, 
                      threshold,num_subchains, K):
    # ==============
    # F stat section
    # ===============
    M = num_subchains/K
    df1 = K-1
    df2 = K*(M-1)


    f_low = f.ppf(0.05, df1, df2)
    f_high = f.ppf(0.95, df1, df2)

    rhat_low = np.sqrt(1+(f_low)/M)-1
    rhat_high = np.sqrt(1+(f_high)/M)-1

    # ===============
    # scatterplots
    # ===============
    fig, axes = plt.subplots(2,3, figsize=(25,10),dpi=150,
                             sharex=True, sharey = True)
    axes = axes.flatten()

    warmups = np.sort(
        np.unique(np.concatenate(
            [df["Warmup Length"].unique() for df in dfs])))
    cmap = plt.get_cmap("coolwarm")
    color_map = {
        w:cmap(x)
        for w, x in zip(warmups, np.linspace(0,1,len(warmups)))} 
    handles = [
        Line2D([0],[0],marker="o", color = color_map[warmup],
               linestyle = "", markersize = 7, label=str(warmup)) for warmup in warmups]

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
    fig.legend(
        handles = handles,
        title = "Warmup Length",
        loc = "center left",
        bbox_to_anchor = (0.84, 0.5),
        fontsize = 9,
        title_fontsize = 10,
        ncol=1,
        frameon=True)

    for ax in axes:
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