def plot_mse_rhat_and_contours(
    dfs,
    bound,
    threshold,
    num_subchains,
    K,
    var1,
    var2,
    mean=(0, 0),
    proposal_mean=(2, 2),
    proposal_var1=100,
    proposal_var2=100,
    supertitle="MSE vs. R-hat and Gaussian Contours",
    padding=0.05,
):
    """
    Create a 3x4 figure:
        Rows 1-2: MSE vs. Rhat - 1 scatterplots for eight experiments.
        Row 3: Gaussian target and initialization contours, with
               positive- and negative-correlation targets overlaid.

    Parameters
    ----------
    dfs : list of 8 pandas DataFrames
        Ordered as:
        [rho=0, rho=0.3, rho=0.6, rho=0.9,
         rho=0, rho=-0.3, rho=-0.6, rho=-0.9]

    bound : sequence of length 2
        Horizontal MSE reference lines.

    threshold : float
        Empirical Rhat - 1 threshold.

    num_subchains : int
        Total number of short chains.

    K : int
        Number of superchains.

    var1, var2 : float
        Marginal variances of the target Gaussian distribution.

    mean : sequence of length 2
        Mean of the target Gaussian distribution.

    proposal_mean : sequence of length 2
        Mean of the initialization Gaussian distribution.

    proposal_var1, proposal_var2 : float
        Marginal variances of the initialization distribution.

    supertitle : str
        Overall figure title.

    padding : float
        Proportional padding added to the contour plot limits.
    """

    # ============================================================
    # 1. Validate inputs
    # ============================================================
    if len(dfs) != 8:
        raise ValueError("dfs must contain exactly 8 DataFrames.")

    if var1 <= 0 or var2 <= 0:
        raise ValueError("var1 and var2 must be positive.")

    if proposal_var1 <= 0 or proposal_var2 <= 0:
        raise ValueError("Initialization variances must be positive.")

    if padding < 0:
        raise ValueError("padding must be non-negative.")

    if K <= 0 or num_subchains <= 0:
        raise ValueError("K and num_subchains must be positive.")

    if len(bound) != 2:
        raise ValueError("bound must contain exactly two values.")

    mean = np.asarray(mean, dtype=float)
    proposal_mean = np.asarray(proposal_mean, dtype=float)

    if mean.shape != (2,) or proposal_mean.shape != (2,):
        raise ValueError("Both means must contain exactly two values.")

    # ============================================================
    # 2. F-derived Rhat boundaries
    # ============================================================
    M = num_subchains / K

    if M <= 1:
        raise ValueError("num_subchains / K must be greater than 1.")

    df1 = K - 1
    df2 = K * (M - 1)

    f_low = f.ppf(0.05, df1, df2)
    f_high = f.ppf(0.95, df1, df2)

    rhat_low = np.sqrt(1 + f_low / M) - 1
    rhat_high = np.sqrt(1 + f_high / M) - 1

    # ============================================================
    # 3. Scatterplot configuration
    # ============================================================
    scatter_titles = [
        r"Constrained - Diagonal",
        r"Constrained - $\rho=0.3$",
        r"Constrained - $\rho=0.6$",
        r"Constrained - $\rho=0.9$",
        r"Constrained - Diagonal",
        r"Constrained - $\rho=-0.3$",
        r"Constrained - $\rho=-0.6$",
        r"Constrained - $\rho=-0.9$",
    ]

    # Collect all warmup lengths so colors are consistent across panels.
    warmups = np.sort(
        np.unique(
            np.concatenate([
                df["Warmup Length"].unique()
                for df in dfs
            ])
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

    warmup_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color=color_map[w],
            linestyle="",
            markersize=7,
            label=str(w),
        )
        for w in warmups
    ]

    # ============================================================
    # 4. Contour configuration
    # ============================================================
    probabilities = [0.68, 0.95, 0.99]

    contour_levels = {
        p: chi2.ppf(p, df=2)
        for p in probabilities
    }

    line_styles = {
        0.68: "-",
        0.95: "--",
        0.99: ":",
    }

    target_color = "royalblue"
    negative_target_color = "teal"
    proposal_color = "darkorange"

    # Target covariance matrices have the same marginal variances
    # for rho and -rho, so all four panels can share these limits.
    q99 = contour_levels[0.99]

    target_x_extent = np.sqrt(q99 * var1)
    target_y_extent = np.sqrt(q99 * var2)

    proposal_x_extent = np.sqrt(q99 * proposal_var1)
    proposal_y_extent = np.sqrt(q99 * proposal_var2)

    # Find the largest extent from either distribution and either axis
    max_extent = max(
        abs(mean[0]) + target_x_extent,
        abs(mean[1]) + target_y_extent,
        abs(proposal_mean[0]) + proposal_x_extent,
        abs(proposal_mean[1]) + proposal_y_extent,
    )

    # Add proportional padding
    limit = max_extent * (1 + padding)

    x_min, x_max = -limit, limit
    y_min, y_max = -limit, limit

    # Common grid for every contour panel
    x = np.linspace(x_min, x_max, 500)
    y = np.linspace(y_min, y_max, 500)
    X, Y = np.meshgrid(x, y)

    # Initialization distribution: fixed across all panels.
    Sigma_proposal = np.diag([
        proposal_var1,
        proposal_var2,
    ])
    inv_proposal = np.linalg.inv(Sigma_proposal)

    dx_proposal = X - proposal_mean[0]
    dy_proposal = Y - proposal_mean[1]

    Q_proposal = (
        inv_proposal[0, 0] * dx_proposal**2
        + 2 * inv_proposal[0, 1]
        * dx_proposal * dy_proposal
        + inv_proposal[1, 1] * dy_proposal**2
    )

    dx_target = X - mean[0]
    dy_target = Y - mean[1]

    # ============================================================
    # 5. Create 3x4 figure
    # ============================================================
    fig, axes = plt.subplots(
        3, 4,
        figsize=(30, 19),
        dpi=300,
    )

    # Before the scatterplot loop:
    all_rhat_minus_1 = np.concatenate([
        np.asarray([float(v) - 1 for v in df["Rhat"]])
        for df in dfs
    ])

    # Keep only positive, finite values because the x-axis is logarithmic.
    valid_x = all_rhat_minus_1[
        np.isfinite(all_rhat_minus_1) & (all_rhat_minus_1 > 0)
    ]

    scatter_x_min = valid_x.min()/1.05
    scatter_x_max = valid_x.max()*1.05

    # Collect MSE values across all datasets
    all_mse = np.concatenate([
        np.asarray([float(v) for v in df["MSE"]], dtype=float)
        for df in dfs
    ])

    # Keep only positive, finite values for the logarithmic y-axis
    valid_y = all_mse[
        np.isfinite(all_mse) & (all_mse > 0)
    ]

    if valid_y.size == 0:
        raise ValueError(
            "No finite positive MSE values found for the log y-axis."
        )

    scatter_y_min = valid_y.min()/1.05
    scatter_y_max = valid_y.max()*1.05

    # ============================================================
    # 6. Scatterplots: first two rows
    # ============================================================
    for i, (df, panel_title) in enumerate(
        zip(dfs, scatter_titles)
    ):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        for warmup, subset in df.groupby("Warmup Length"):
            # Convert JAX scalar values to ordinary floats if needed.
            rhat_values = np.asarray(
                [float(v) for v in subset["Rhat"]]
            )
            mse_values = np.asarray(
                [float(v) for v in subset["MSE"]]
            )

            ax.scatter(
                rhat_values - 1,
                mse_values,
                color=color_map[warmup],
                s=28,
                alpha=0.55,
            )

        ax.set_title(
            panel_title,
            fontsize=13,
            pad=8,
        )

        ax.set_xscale("log")
        ax.set_xlim(scatter_x_min, scatter_x_max)
        ax.set_yscale("log")
        ax.set_ylim(scatter_y_min, scatter_y_max)

        # Horizontal MSE bounds.
        for b in bound:
            ax.axhline(
                b,
                color="grey",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )

        # Reference MSE = 1 / number of subchains.
        ax.axhline(
            1 / num_subchains,
            color="black",
            linewidth=1.0,
            alpha=0.7,
        )

        # Empirical Rhat threshold.
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )

        # F-derived Rhat boundaries.
        ax.axvline(
            rhat_low,
            color="purple",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.axvline(
            rhat_high,
            color="purple",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )

        ax.tick_params(
            axis="both",
            which="both",
            labelsize=10,
        )

        ax.grid(
            which="major",
            linestyle="--",
            linewidth=0.5,
            alpha=0.3,
        )

        ax.set_xlabel(r"$\widehat{R}-1$", fontsize=11)
        ax.set_ylabel("Scaled Squared Error", fontsize=11)

    # ============================================================
    # 7. Contours: third row
    # ============================================================
    # Each panel compares:
    #   * Initialization distribution (orange)
    #   * Positive-correlation target (blue)
    #   * Negative-correlation target (teal), when applicable
    contour_rhos = [0, 0.3, 0.6, 0.9]

    for j, rho in enumerate(contour_rhos):
        ax = axes[2, j]

        # Initialization contours.
        for p in probabilities:
            ax.contour(
                X,
                Y,
                Q_proposal,
                levels=[contour_levels[p]],
                colors=[proposal_color],
                linestyles=[line_styles[p]],
                linewidths=1.5,
            )

        # Positive or diagonal target covariance.
        Sigma_target_pos = np.array([
            [var1, rho * np.sqrt(var1 * var2)],
            [rho * np.sqrt(var1 * var2), var2],
        ])

        inv_target_pos = np.linalg.inv(Sigma_target_pos)

        Q_target_pos = (
            inv_target_pos[0, 0] * dx_target**2
            + 2 * inv_target_pos[0, 1]
            * dx_target * dy_target
            + inv_target_pos[1, 1] * dy_target**2
        )

        for p in probabilities:
            ax.contour(
                X,
                Y,
                Q_target_pos,
                levels=[contour_levels[p]],
                colors=[target_color],
                linestyles=[line_styles[p]],
                linewidths=1.7,
            )

        # Overlay the corresponding negative-correlation target.
        # There is no separate negative target for rho = 0.
        if rho > 0:
            Sigma_target_neg = np.array([
                [var1, -rho * np.sqrt(var1 * var2)],
                [-rho * np.sqrt(var1 * var2), var2],
            ])

            inv_target_neg = np.linalg.inv(Sigma_target_neg)

            Q_target_neg = (
                inv_target_neg[0, 0] * dx_target**2
                + 2 * inv_target_neg[0, 1]
                * dx_target * dy_target
                + inv_target_neg[1, 1] * dy_target**2
            )

            for p in probabilities:
                ax.contour(
                    X,
                    Y,
                    Q_target_neg,
                    levels=[contour_levels[p]],
                    colors=[negative_target_color],
                    linestyles=[line_styles[p]],
                    linewidths=1.7,
                )

        # Panel titles.
        if rho == 0:
            panel_title = r"Target $\rho=0$"
        else:
            panel_title = (
                rf"Target $\rho=+{rho}$ and $\rho=-{rho}$"
            )

        ax.set_title(
            panel_title,
            fontsize=13,
            pad=8,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(r"$x_1$", fontsize=11)
        ax.set_ylabel(r"$x_2$", fontsize=11)

        ax.axhline(
            0,
            color="grey",
            linestyle="--",
            linewidth=0.7,
            alpha=0.35,
        )
        ax.axvline(
            0,
            color="grey",
            linestyle="--",
            linewidth=0.7,
            alpha=0.35,
        )

        ax.grid(
            which="major",
            linestyle="--",
            linewidth=0.5,
            alpha=0.25,
        )

        ax.tick_params(
            axis="both",
            which="both",
            labelsize=10,
        )

    # ============================================================
    # 8. Legends
    # ============================================================
    # Warmup-length legend for scatterplots.
    fig.legend(
        handles=warmup_handles,
        title="Warmup Length",
        loc="upper left",
        bbox_to_anchor=(0.865, 0.88),
        fontsize=9,
        title_fontsize=10,
        frameon=True,
    )

    # Distribution colors and probability line styles for contours.
    contour_handles = [
        Line2D(
            [0], [0],
            color=target_color,
            linewidth=1.8,
            label="Target: positive / zero rho",
        ),
        Line2D(
            [0], [0],
            color=negative_target_color,
            linewidth=1.8,
            label="Target: negative rho",
        ),
        Line2D(
            [0], [0],
            color=proposal_color,
            linewidth=1.8,
            label="Initialization",
        ),
    ]

    for p in probabilities:
        contour_handles.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle=line_styles[p],
                linewidth=1.8,
                label=f"{p:.0%} contour",
            )
        )

    fig.legend(
        handles=contour_handles,
        title="Contour Legend",
        loc="upper left",
        bbox_to_anchor=(0.865, 0.48),
        fontsize=9,
        title_fontsize=10,
        frameon=True,
    )

    # ============================================================
    # 9. Overall title and layout
    # ============================================================
    fig.suptitle(
        supertitle,
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.02, 0.955,
        f"K = {K}, M = {M:g}",
        fontsize=12,
        ha="left",
        va="top",
    )

    fig.text(
        0.02, 0.937,
        rf"Target variances: $\sigma_1^2={var1},"
        rf"\quad \sigma_2^2={var2}$",
        fontsize=12,
        ha="left",
        va="top",
    )

    fig.subplots_adjust(
        top=0.90,
        bottom=0.06,
        left=0.055,
        right=0.85,
        hspace=0.38,
        wspace=0.30,
    )

    plt.show()