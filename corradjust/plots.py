import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from adjustText import adjust_text
import numpy as np

from corradjust.utils import *


class PCAVariancePlotter:
    """
    Create PCA explained variance plot.

    Parameters
    ----------
    plot_width : float, optional, default=6.4
        Plot width.
    plot_height : float, optional, default=4.8
        Plot height.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
    axs : dict
        Keys of `axs` are strings ``"individual"`` and
        ``"cumulative"`` referring to the two panels of the plot.
        Values of `axs` are instances of `matplotlib.axes.Axes`.
    """

    def __init__(self, plot_width=6.4, plot_height=4.8):
        fig, axs = plt.subplots(1, 2, figsize=(plot_width * 2, plot_height))
        axs = {"individual": axs[0], "cumulative": axs[1]}

        axs["individual"].set_xlabel("PC")
        axs["individual"].set_ylabel("% variance (individual)")

        axs["cumulative"].set_xlabel("PC")
        axs["cumulative"].set_ylabel("% variance (cumulative)")
        axs["cumulative"].set_ylim([0, 105])

        self.fig = fig
        self.axs = axs
    
    def plot(self, PCA_model, n_PCs):
        """
        Draw the plots.

        Parameters
        ----------
        PCA_model : sklearn.decomposition.PCA
            PCA model. The `fit` method should be
            called on `PCA_model` prior to calling this method.
        n_PCs : int
            Number of PCs to plot on the X-axis.
        """

        vars = PCA_model.explained_variance_ratio_ * 100
        comps = np.arange(1, len(vars) + 1)

        sns.lineplot(
            x=comps, y=vars,
            marker="o", markeredgewidth=0,
            ax=self.axs["individual"]
        )
        self.axs["individual"].set_ylim([0, np.max(vars) + 5])

        sns.lineplot(
            x=comps, y=np.cumsum(vars),
            marker="o", markeredgewidth=0,
            ax=self.axs["cumulative"]
        )
        self.axs["cumulative"].axvline(
            n_PCs, ls=":", color="red",
            label=f"Knee (n={n_PCs})"
        )
    
    def save_plot(self, out_path, title=None):
        """
        Save the plot. This method doesn't
        call `plt.close`, so it will display the figure in
        jupyter notebook in addition to saving the file.

        Parameters
        ----------
        out_path : str
            Path to the figure (with extension, e.g., ``".png"``).
        title : str or None, optional, default=None
            Short text to show at the top-left corner of the plot.
        """

        self.axs["cumulative"].legend(loc="lower right")
        
        if title:
            self.fig.text(0.01, 0.99, title, va="top", transform=self.fig.transFigure)
            self.fig.tight_layout(rect=[0, 0, 1, 0.98])
        else:
            self.fig.tight_layout()

        self.fig.savefig(out_path, dpi=300)


class GreedyOptimizationPlotter:
    """
    Create a lineplot with score optimization trajectories.

    Parameters
    ----------
    samp_group_name : str or None, optional, default=None
        Main title with sample group name to use in the plot.
    metric : {"enrichment-based", "BP@K"}, optional, default="enrichment-based"
        Metric for evaluating feature correlations.
    palette : str or list or dict, optional, default="Dark2"
        Name of matplotlib colormap or list of colors for the lines or
        dict mapping reference collection names to colors.
        The argument is directly passed to `sns.lineplot`.
    legend_loc : str, optional, default="lower right"
        Where to put the legend (follows matplotlib notation).
    legend_fontsize : int, optional, default=10
        Font size of legend text.
    plot_width : float, optional, default=6.4
        Plot width.
    plot_height : float, optional, default=4.8
        Plot height.
    
    Attributes
    ----------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    def __init__(
        self,
        samp_group_name=None,
        metric="enrichment-based",
        palette="Dark2",
        legend_loc="lower right",
        legend_fontsize=10,
        plot_width=6.4,
        plot_height=4.8
    ):

        # Now create a figure for greedy optimization
        fig, ax = plt.subplots(1, 1, figsize=(plot_width, plot_height))
       
        ax.set_xlabel("Iteration")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if metric == "enrichment-based":
            ax.set_ylabel("Average -log$_{{10}}$(adjusted p-value)")
        else:
            ax.set_ylabel("Balanced precision at K")

        if samp_group_name:
            ax.set_title(samp_group_name)
        
        self.palette = palette
        self.legend_loc = legend_loc
        self.legend_fontsize = legend_fontsize

        self.fig = fig
        self.ax = ax

    def plot(self, df, peak_iter):
        """
        Draw the lines.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame with scores. One can generate
            `df` by taking ``fit.tsv`` table and
            selecting columns for only one sample group.
            See `CorrAdjust._make_best_iter_scores_plot``
            method for an example.
        peak_iter : int
            This number is used to draw vertical dashed red
            line at the selected early stopping iteration.
        """

        self.peak_iter = peak_iter

        # Remove sample group name from the columns
        df = df.rename(columns={col: ";".join(col.split(";")[1:]) for col in df.columns})
        
        # Plot is either called with training and validation (pairs)
        # or with training and test (samples)
        if "mean;validation" in set(df.columns):
            subset_label = "Feature pairs"
            self.validation_pairs = True
            # 3 columns per reference collection (training, validation, all)
            self.num_ref_feature_colls = len(df.columns) // 3
            # Plot only training and validation
            df = df.drop(columns=[
                col for col in df.columns
                if not col.endswith(";training") and not col.endswith(";validation")
            ])
        else:
            subset_label = "Samples"
            # 2 columns per reference collection (training, test)
            self.num_ref_feature_colls = len(df.columns) // 2
            # Plot only training and test
            df = df.drop(columns=[
                col for col in df.columns
                if not col.endswith(";training") and not col.endswith(";test")
            ])

        # Re-order columns so that mean goes first (it is last by default)
        df = df[df.columns[-2:].tolist() + df.columns[:-2].tolist()]
        
        # We don't plot mean if there is only 1 reference collection
        # self.num_ref_feature_colls already includes "mean" as a collection
        if self.num_ref_feature_colls == 2:
            df = df.iloc[:, 2:]
            self.num_ref_feature_colls = 1
        
        df = df.melt(var_name="metric_name", value_name="score", ignore_index=False)
        df = df.reset_index()
        df["Ref. collection"] = df["metric_name"].str.split(";").str[0]
        # Capitalize word "mean"
        df["Ref. collection"] = df["Ref. collection"].str.replace("mean", "Mean")
        
        df[subset_label] = df["metric_name"].str.split(";").str[-1].str.capitalize()
        sns.lineplot(
            x="Iteration", y="score", hue="Ref. collection", style=subset_label, data=df,
            palette=self.palette,
            dashes=["", (1, 1)],
            markers=["o", "X"], markeredgewidth=0, markersize=4,
            ax=self.ax
        )
        
        # Set highest zorder for the first drawn lines
        zorder = 1000
        for line in self.ax.get_lines():
            line.set_zorder(zorder)
            zorder -= 1

        x_pad = df["Iteration"].max() * 0.02
        self.ax.set_xlim(-x_pad, df["Iteration"].max() + x_pad)

        y_pad = df["score"].max() * 0.02
        if df["score"].max() != 0:
            self.ax.set_ylim(-y_pad, df["score"].max() + y_pad)
        else:
            self.ax.set_ylim(-0.01, 1.02)

    def save_plot(self, out_path, title=None):
        """
        Save the plot. This method doesn't
        call `plt.close`, so it will display the figure in
        jupyter notebook in addition to saving the file.

        Parameters
        ----------
        out_path : str
            Path to the figure (with extension, e.g., ``.png``).
        title : str or None, optional, default=None
            Short text to show at the top-left corner of the plot.
        """

        # We draw this here to make legend code easier
        self.ax.axvline(
            self.peak_iter, ls=":", color="red",
            label=f"Iter = {self.peak_iter}"
        )

        # Make a legend
        handles, labels = self.ax.get_legend_handles_labels()
        new_handles, new_labels = [], []
        empty_handle = mpatches.Patch(color="none")
        empty_label = " "

        # Number of rows in the last column
        n_last = 5
        
        nrows = max(self.num_ref_feature_colls + 1, n_last)
        # No more than 5 rows per column
        nrows = min(nrows, 5)

        # For reference collections, each column has a header
        # So, we have 4 collections per column
        ncols = self.num_ref_feature_colls // (nrows - 1)
        if self.num_ref_feature_colls % (nrows - 1):
            ncols += 1
        
        # Last column is always for training, validation, and early stopping
        ncols += 1

        # Legend for reference sets
        # First, fill in complete columns
        complete_columns = self.num_ref_feature_colls // (nrows - 1)
        for i in range(complete_columns):
            new_handles += [handles[0]]
            new_labels += [labels[0]]
            
            new_handles += handles[1 + (nrows - 1) * i : 1 + (nrows - 1) * (i + 1)]
            new_labels += labels[1 + (nrows - 1) * i : 1 + (nrows - 1) * (i + 1)]

        # An additional incomplete column
        if self.num_ref_feature_colls % (nrows - 1):
            new_handles += [handles[0]]
            new_labels += [labels[0]]
            
            new_handles += handles[1 + (nrows - 1) * complete_columns:1 + self.num_ref_feature_colls]
            new_labels += labels[1 + (nrows - 1) * complete_columns:1 + self.num_ref_feature_colls]

            new_handles += [empty_handle] * (nrows - 1 - self.num_ref_feature_colls % (nrows - 1))
            new_labels += [empty_label] * (nrows - 1 - self.num_ref_feature_colls % (nrows - 1))

        # Last column
        new_handles += handles[self.num_ref_feature_colls + 1:self.num_ref_feature_colls + 1 + 3]
        new_labels += labels[self.num_ref_feature_colls + 1:self.num_ref_feature_colls + 1 + 3]
        new_handles += [empty_handle] + handles[-1:]
        new_labels += ["Early stopping"] + labels[-1:]

        legend = self.ax.legend(
            new_handles, new_labels, ncols=ncols,
            loc=self.legend_loc, fontsize=self.legend_fontsize
        )
        legend.set_zorder(2000)
        
        # Remove left padding for legend titles
        # first loop goes over legend columns
        for i, vpack in enumerate(legend._legend_handle_box.get_children()):
            for j, hpack in enumerate(vpack.get_children()):
                if j == 0 or (i == ncols - 1 and j == 3):
                    hpack.get_children()[0].set_width(0)

        if title:
            self.fig.text(0.01, 0.99, title, va="top", transform=self.fig.transFigure)
            self.fig.tight_layout(rect=[0, 0, 1, 0.98])
        else:
            self.fig.tight_layout()

        self.fig.savefig(out_path, dpi=300)


class VolcanoPlotter:
    """
    Create a volcano plot with feature-wise enrichment statistics.

    Parameters
    ----------
    corr_scorer : CorrScorer
        Instance of `CorrScorer`.
    annotate_features : int or None, optional, default=None
        How many features with the lowest adjusted p-value to annotate.
    annot_fontsize : int, optional, default=8
        Font size of feature names when ``annotate_features=True``.
    feature_name_fmt : function or None, optional, default=None
        Function that maps feature names to labels to show on the plot
        when ``annotate_features=True``. If ``None``, shows unmodified feature names.
    signif_color : matplotlib color, optional
        Color to draw statistically significant features.
    nonsignif_color : matplotlib color, optional
        Color to draw non-significant features.
    panel_size : float, optional, default=4.8
        Size of each (square) panel.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
    axs : dict
        Keys of `axs` are tuples ``(ref_feature_coll, state)``,
        where ``state`` is either ``"Raw"`` or ``"Clean"``.
        Values of `axs` are instances of `matplotlib.axes.Axes`.
    """

    def __init__(
        self,
        corr_scorer,
        annotate_features=False,
        annot_fontsize=8,
        feature_name_fmt=None,
        signif_color=(*sns.color_palette("Set1")[0], 0.9),
        nonsignif_color=(0.6, 0.6, 0.6, 0.5),
        panel_size=4.8
    ):
        n_rows = len(corr_scorer.data)
        n_columns = 2
        fig, axs = plt.subplots(n_rows, n_columns, figsize=(panel_size * n_columns, panel_size * n_rows))
        
        # By default, axs object is 2D array
        # We convert it to dict to have meaningful keys
        axs_dict = {}
        for i, ref_feature_coll in enumerate(corr_scorer.data):
            for j, state in enumerate(["Raw", "Clean"]):
                if len(corr_scorer.data) > 1:
                    axs_dict[(ref_feature_coll, state)] = axs[i, j]
                else:
                    axs_dict[(ref_feature_coll, state)] = axs[j]

        axs = axs_dict
        
        for ref_feature_coll in corr_scorer.data:
            for state in ["Raw", "Clean"]:
                axs[(ref_feature_coll, state)].set_title(f"{ref_feature_coll}, {state.lower()} corr.")
                
                axs[(ref_feature_coll, state)].set_xlabel("Balanced precision")
                axs[(ref_feature_coll, state)].set_xlim([-0.03, 1.03])
                axs[(ref_feature_coll, state)].set_xticks(np.arange(0, 1.1, 0.1))

                axs[(ref_feature_coll, state)].set_ylabel("$-$log$_{{10}}$(adjusted p-value)")
        
        self.fig = fig
        self.axs = axs
        self.annotate_features = annotate_features
        self.annot_fontsize = annot_fontsize
        self.feature_name_fmt = feature_name_fmt
        self.annotations = {}
        self.metric = corr_scorer.metric

        self.signif_color = signif_color
        self.nonsignif_color = nonsignif_color

        self.min_marker_size = 10 # seaborn default 18
        self.max_marker_size = 100 # seaborn default 72

    def plot(self, feature_scores):
        """
        Create a volcano plot.

        Parameters
        ----------
        feature_scores : dict
            Dict with feature-wise enrichment scores generated
            by the `CorrAdjust.compute_feature_scores` method.
        """

        # We need to have identical y-axis limits and legends for numbers of
        # pairs beterrn raw and clean corrs
        num_pairs_max = {}
        for ref_feature_coll in feature_scores["Raw"]:
            df_raw = feature_scores["Raw"][ref_feature_coll].dropna()
            padj_raw = -np.log10(df_raw["padj"])
            num_pairs_raw = df_raw["ref_pairs@K"].str.split("/").str[1].astype("int64")
            
            if feature_scores["Clean"] is not None:
                df_clean = feature_scores["Clean"][ref_feature_coll].dropna()
                padj_clean = -np.log10(df_clean["padj"])
                num_pairs_clean = df_clean["ref_pairs@K"].str.split("/").str[1].astype("int64")

                padj_max = max(np.max(padj_raw), np.max(padj_clean))
                num_pairs_max[ref_feature_coll] = max(np.max(num_pairs_raw), np.max(num_pairs_clean))
            else:
                padj_max = np.max(padj_raw)
                num_pairs_max[ref_feature_coll] = np.max(num_pairs_raw)

            if padj_max != 0:
                y_pad = padj_max * 0.03 # So the points won't stick to ax bottom/top
            else:
                y_pad = 1 

            for state in ["Raw", "Clean"]:
                self.axs[(ref_feature_coll, state)].set_ylim([-y_pad, padj_max + y_pad])

                self.axs[(ref_feature_coll, state)].fill_between(
                    [-0.03, 0.5], [padj_max + y_pad, padj_max + y_pad], [-y_pad, -y_pad],
                    facecolor=(*sns.color_palette("tab10")[0], 0.07),
                    zorder=0
                )
                self.axs[(ref_feature_coll, state)].fill_between(
                    [0.5, 1.03], [padj_max + y_pad, padj_max + y_pad], [-y_pad, -y_pad],
                    facecolor=(*sns.color_palette("tab10")[3], 0.07),
                    zorder=0
                )
        
        self._plot_one_state("Raw", feature_scores["Raw"], num_pairs_max)
        if feature_scores["Clean"] is not None:
            self._plot_one_state("Clean", feature_scores["Clean"], num_pairs_max)

    def _plot_one_state(
        self,
        state,
        feature_scores_state,
        num_pairs_max
    ):
        """
        Make the plot for ``"Raw"`` or ``"Clean"`` data.

        Parameters
        ----------
        state : {"Raw", "Clean"}
        feature_scores_state : {feature_scores["Raw"], feature_scores["Clean"]}
        num_pairs_max : int
            Maximum number for ``K_j`` to scale the markers and
            display in legend.
        """

        for ref_feature_coll in feature_scores_state:
            df_plot = feature_scores_state[ref_feature_coll].dropna().copy()
            df_plot["padj"] = -np.log10(df_plot["padj"])
            df_plot["num_pairs"] = df_plot["ref_pairs@K"].str.split("/").str[1].astype("int64")
            
            if self.metric == "enrichment-based":
                avg_log_padj = df_plot["padj"].mean()
                score_label = f"{avg_log_padj:.2f}"
            else:
                agg_BP_at_K, agg_enrichment, agg_pvalue = (
                    compute_aggregated_scores(df_plot)
                )
                score_label = f"{agg_BP_at_K:.2f}"

            perc_signif = (10**(-df_plot["padj"]) <= 0.05).sum() / len(df_plot) * 100
            signif_label = f"Yes ({np.round(perc_signif, 1)}%)"
            nonsignif_label = f"No ({np.round(100 - np.round(perc_signif, 1), 1)}%)"
            df_plot["Adj. p ≤ 0.05"] = [
                signif_label if 10**(-p) <= 0.05 else nonsignif_label
                for p in df_plot["padj"]
            ]
            
            sns.scatterplot(
                x="balanced_precision", y="padj", hue="Adj. p ≤ 0.05", size="num_pairs", data=df_plot,
                hue_order=[signif_label, nonsignif_label],
                palette={
                    signif_label: self.signif_color,
                    nonsignif_label: self.nonsignif_color
                },
                sizes=(self.min_marker_size, self.max_marker_size),
                size_norm=(0, num_pairs_max[ref_feature_coll]),
                ax=self.axs[(ref_feature_coll, state)]
            )
            self.axs[(ref_feature_coll, state)].set_title(
                self.axs[(ref_feature_coll, state)].get_title() +
                f", score = {score_label}"
            )
            
            # Fix legend
            empty_handle = mpatches.Patch(color="none")
            empty_label = " "
            
            handles, labels = self.axs[(ref_feature_coll, state)].get_legend_handles_labels()
            # Keep the legend part about statistical significance
            handles, labels = handles[:3], labels[:3]

            # Empty line
            handles.append(empty_handle)
            labels.append(empty_label)

            # Title
            handles.append(empty_handle)
            labels.append("# highly ranked\nfeature pairs ($K_j$)")

            # Markers
            handles += [
                Line2D(
                    [], [], markersize=np.sqrt(self.min_marker_size), markeredgewidth=1,
                    color="black",  markeredgecolor="white",
                    linestyle="None", marker="o"
                ),
                Line2D(
                    [], [], markersize=np.sqrt(self.max_marker_size), markeredgewidth=1,
                    color="black",  markeredgecolor="white",
                    linestyle="None", marker="o"
                )
            ]
            labels += ["0", str(num_pairs_max[ref_feature_coll])]

            legend = self.axs[(ref_feature_coll, state)].legend(handles, labels, loc="upper left")

            # Remove left padding for legend titles
            # first loop is just 1 iteration since legend is 1 column
            for vpack in legend._legend_handle_box.get_children():
                for j, hpack in enumerate(vpack.get_children()):
                    if j in {0, 4}:
                        hpack.get_children()[0].set_width(0)

            if self.annotate_features:
                # Annotate features with the lowest padj
                df_top_features = df_plot.loc[10**(-df_plot["padj"]) <= 0.05].iloc[:self.annotate_features]
                annotations = []
                for _, row in df_top_features.iterrows():
                    if not self.feature_name_fmt:
                        text = row["feature_name"]
                    else:
                        text = self.feature_name_fmt(row["feature_name"])
                        
                    annotation = self.axs[(ref_feature_coll, state)].text(
                        x=row["balanced_precision"], y=row["padj"], s=text,
                        ha="center", va="center", fontsize=self.annot_fontsize
                    )
                    annotations.append(annotation)
                
                self.annotations[(ref_feature_coll, state)] = annotations

    def save_plot(self, out_path, title=None):
        """
        Save the plot. This method doesn't
        call `plt.close`, so it will display the figure in
        jupyter notebook in addition to saving the file.

        Parameters
        ----------
        out_path : str
            Path to the figure (with extension, e.g., ``.png``).
        title : str or None, optional, default=None
            Short text to show at the top-left corner of the plot.
        """

        if title:
            self.fig.text(0.01, 0.99, title, va="top", transform=self.fig.transFigure)
            self.fig.tight_layout(rect=[0, 0, 1, 0.98])
        else:
            self.fig.tight_layout()

        # adjust_text should be called after everything else is done        
        if self.annotate_features:
            for ref_feature_coll, state in self.annotations:
                adjust_text(
                    self.annotations[(ref_feature_coll, state)], ax=self.axs[(ref_feature_coll, state)],
                    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5, shrinkA=1, shrinkB=1),
                    explode_radius=10
                )

        self.fig.savefig(out_path, dpi=300)


class CorrDistrPlotter:
    """
    Create KDE and CDF plots of correlations.

    Parameters
    ----------
    corr_scorer : CorrScorer
        Instance of `CorrScorer`.
    pairs_subset : {"all", "training", "validation"}, optional, default="all"
        Which set of feature pairs to use for computing scores.
    color_raw_ref : color, optional, default=sns.color_palette("tab20")[0]
        Color for raw correlations, reference feature pairs.
    color_raw_non_ref : color, optional, default=sns.color_palette("tab20")[1],
        Color for raw correlations, non-reference feature pairs.
    color_clean_ref : color, optional, default=sns.color_palette("tab20")[2],
        Color for clean correlations, reference feature pairs.
    color_clean_non_ref : color, optional, default=sns.color_palette("tab20")[3],
        Color for clean correlations, non-reference feature pairs.
    legend_fontsize : int, optional, default=10
        Font size of legend text.
    panel_size : float, optional, default=4.8
        Size of each (square) panel.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
    axs : dict
        Keys of `axs` are tuples ``(ref_feature_coll, plot_name)``,
        where ``plot_name`` is either ``"corr-KDE"`` or ``"corr-CDF"``.
        Values of `axs` are instances of `matplotlib.axes.Axes`.
    """

    def __init__(
        self,
        corr_scorer,
        pairs_subset="all",
        color_raw_ref=sns.color_palette("tab20")[0],
        color_raw_non_ref=sns.color_palette("tab20")[1],
        color_clean_ref=sns.color_palette("tab20")[2],
        color_clean_non_ref=sns.color_palette("tab20")[3],
        legend_fontsize=10,
        panel_size=4.8
    ):
        n_rows = len(corr_scorer.data)
        n_columns = 2
        fig, axs = plt.subplots(n_rows, n_columns, figsize=(panel_size * n_columns, panel_size * n_rows))
        
        # By default, axs object is 2D array
        # We convert it to dict to have meaningful keys
        axs_dict = {}
        for i, ref_feature_coll in enumerate(corr_scorer.data):
            for j, plot_name in enumerate(["corr-KDE", "corr-CDF"]):
                if len(corr_scorer.data) > 1:
                    axs_dict[(ref_feature_coll, plot_name)] = axs[i, j]
                else:
                    axs_dict[(ref_feature_coll, plot_name)] = axs[j]

        axs = axs_dict
        
        for ref_feature_coll in corr_scorer.data:
            sign = corr_scorer.data[ref_feature_coll]["sign"]
            if sign == "absolute":
                corr_label = "Absolute correlation"
                corr_lim = [-0.05, 1.05]
            else:
                corr_label = "Correlation"
                corr_lim = [-1.05, 1.05]
            
            high_corr_frac = corr_scorer.data[ref_feature_coll]["high_corr_frac"]
            
            axs[(ref_feature_coll, "corr-KDE")].set_title(ref_feature_coll)
            axs[(ref_feature_coll, "corr-KDE")].set_xlabel("Correlation")
            axs[(ref_feature_coll, "corr-KDE")].set_xlim(-1.05, 1.05)
            axs[(ref_feature_coll, "corr-KDE")].set_ylabel("Density")

            axs[(ref_feature_coll, "corr-CDF")].set_title(ref_feature_coll)
            axs[(ref_feature_coll, "corr-CDF")].set_xlabel(corr_label)
            axs[(ref_feature_coll, "corr-CDF")].set_xlim(*corr_lim)
            axs[(ref_feature_coll, "corr-CDF")].set_ylabel(f"Cumulative fraction of feature pairs")
            axs[(ref_feature_coll, "corr-CDF")].set_yscale("log")
            axs[(ref_feature_coll, "corr-CDF")].axhline(
                high_corr_frac, ls=":", color="grey", label=f"Highly ranked pairs"
            )
        
        self.legend_fontsize = legend_fontsize
        self.colors = {
            ("Raw", 0): color_raw_non_ref,
            ("Raw", 1): color_raw_ref,
            ("Clean", 0): color_clean_non_ref,
            ("Clean", 1): color_clean_ref
        }

        self.fig = fig
        self.axs = axs

        self.pairs_subset = pairs_subset
        self.signs = {
            ref_feature_coll: corr_scorer.data[ref_feature_coll]["sign"]
            for ref_feature_coll in corr_scorer.data
        }

    def add_plots(
        self,
        corr_scores,
        state,
        num_points=100000
    ):
        """
        Make KDE and eCDF plots.

        Parameters
        ----------
        corr_scores : dict
            Results of `CorrAdjust.compute_corr_scores` method.
        state : {"Raw", "Clean"}
        num_points : int, optional, default=100000
            How many correlations to sample for plotting.
        """

        for ref_feature_coll in corr_scores:
            corrs = corr_scores[ref_feature_coll]["corrs"]
            mask = corr_scores[ref_feature_coll]["mask"]
            train_val_mask = corr_scores[ref_feature_coll]["train_val_mask"]

            # Limit to training/validation pairs if needed
            if self.pairs_subset == "training":
                corrs = corrs[train_val_mask == 0]
                mask = mask[train_val_mask == 0]
            elif self.pairs_subset == "validation":
                corrs = corrs[train_val_mask == 1]
                mask = mask[train_val_mask == 1]

            for ref_flag in [1, 0]:
                corrs_subset = corrs[mask == ref_flag]
                color = self.colors[(state, ref_flag)]
                linestyle = "-" if ref_flag == 1 else "--"

                # Make KDE plot

                # Downsample corrs to allow plotting in adequate time
                if len(corrs_subset) >= num_points:
                    idx_grid = np.linspace(0, len(corrs_subset) - 1, num=num_points, dtype=int)
                    # dtype=int might cause duplicates because of rounding - kill them
                    idx_grid = np.unique(idx_grid)
                    corrs_for_KDE = corrs_subset[idx_grid]
                else:
                    corrs_for_KDE = corrs_subset
                
                sns.kdeplot(
                    x=corrs_for_KDE,
                    label=f"{state}, {'ref. pairs' if ref_flag == 1 else 'non-ref. pairs'}",
                    color=color, linestyle=linestyle,
                    ax=self.axs[(ref_feature_coll, "corr-KDE")]
                ) 
                
                # Make CDF plot

                ranks = np.arange(1, corrs_subset.shape[0] + 1)
                fractions = ranks / np.max(ranks)
                # Downsample corrs to allow plotting in adequate time
                # Since Y axis is log, we use logspace of indices
                if len(corrs_subset) >= num_points:
                    idx_grid = np.logspace(0, np.log10(len(corrs_subset)), num=num_points, base=10, dtype=int) - 1
                    # dtype=int might cause duplicates because of rounding - kill them
                    idx_grid = np.unique(idx_grid)
                    corrs_for_CDF = corrs_subset[idx_grid]
                    fractions = fractions[idx_grid]
                else:
                    corrs_for_CDF = corrs_subset
                
                if self.signs[ref_feature_coll] == "absolute":
                    corrs_for_CDF = np.abs(corrs_for_CDF)
                
                self.axs[(ref_feature_coll, "corr-CDF")].plot(
                    corrs_for_CDF, fractions,
                    label=f"{state}, {'ref. pairs' if ref_flag == 1 else 'non-ref. pairs'}",
                    color=color, linestyle=linestyle
                )

    def save_plot(self, out_path, title=None):
        """
        Save the plot. This method doesn't
        call `plt.close`, so it will display the figure in
        jupyter notebook in addition to saving the file.

        Parameters
        ----------
        out_path : str
            Path to the figure (with extension, e.g., ``.png``).
        title : str or None, optional, default=None
            Short text to show at the top-left corner of the plot.
        """

        for (ref_feature_coll, plot_name), ax in self.axs.items():
            sign = self.signs[ref_feature_coll]
            handles, labels = ax.get_legend_handles_labels()

            if plot_name == "corr-CDF":
                # Put K last
                handles = handles[1:] + [handles[0]]
                labels = labels[1:] + [labels[0]]
                loc = "lower right" if sign == "negative" else "lower left"
            
            if plot_name == "corr-KDE":
                loc = "upper left"

            legend = ax.legend(handles, labels, loc=loc, fontsize=self.legend_fontsize)

            # For KDE plot, we increase ylim to fit the legend on top
            if plot_name == "corr-KDE":
                bbox = legend.get_window_extent()
                legend_frac = bbox.transformed(ax.transAxes.inverted()).height
                y_min, y_max = ax.get_ylim()
                y_max_new = (y_max - y_min) / (1 - legend_frac)
                ax.set_ylim([-y_max_new * 0.01, y_max_new])

        if title:
            self.fig.text(0.01, 0.99, title, va="top", transform=self.fig.transFigure)
            self.fig.tight_layout(rect=[0, 0, 1, 0.98])
        else:
            self.fig.tight_layout()
        
        self.fig.savefig(out_path, dpi=300)