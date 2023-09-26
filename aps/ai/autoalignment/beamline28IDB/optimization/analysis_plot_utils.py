import numpy as np
import cmasher as cmm
import matplotlib as mpl
from matplotlib.axes import Axes
from aps.ai.autoalignment.common.util.common import Histogram, DictionaryWrapper
import dataclasses as dt
from optuna import Study, Trial


#CMAP = cmm.get_sub_cmap(cmm.sunburst_r, 0, 0.5)
#CMAP.set_bad("w")
CMAP = cmm.sunburst_r
PARETO_PLOT_ORDER = {"FWHM": 0, "PL": 1, "NLPI": 2}


@dt.dataclass
class AnalyzedProps:
    study: Study
    nash_trial: Trial
    max_count: int
    min_count: int = 200
    xylim: float = 0.15
    photon_count_str: str = "3.1e"
    ntrials: int = dt.field(init=False)
    ns: list = dt.field(init=False)
    pf: list = dt.field(init=False)
    other: list = dt.field(init=False)

    def __post_init__(self):
        self.ntrials = len(self.study.trials)
        self.ns = [self.nash_trial.number]
        self.pf = [t.number for t in self.study.best_trials if t.number != self.nash_trial.number]
        self.other = [n for n in range(len(self.study.trials)) if n not in self.ns and n not in self.pf]


def plot_hist_2d(
    props: AnalyzedProps,
    ax: Axes,
    hist: Histogram,
    labelfontsize: int = 12,
    ticks: list = None,
    sublabel: str = "a",
    sublabel_kwargs: dict = None,
    study_num: int = None,
    study_num_kwargs: dict = None,
    ylabel: bool = False,
):
    # norm = mpl.colors.LogNorm(props.min_count, props.max_count)
    norm = mpl.colors.Normalize(0, props.max_count)
    cmesh = ax.pcolormesh(hist.hh, hist.vv[::-1], hist.data_2D.T, cmap=CMAP, rasterized=True)# norm=norm, rasterized=True)

    ax.set_aspect("equal")

    lims = (-props.xylim, props.xylim)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.axhline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
    ax.axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)

    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=8)
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.set_xlabel("x (mm)", fontsize=labelfontsize)

    kwargs1 = {
        "x": 0.02,
        "y": 0.9,
        "s": f"{sublabel})",
        "transform": ax.transAxes,
        "size": 16,
        "color": "red",
        "fontweight": "bold",
    }
    if sublabel_kwargs is not None:
        kwargs1.update(sublabel_kwargs)

    ax.text(**kwargs1)

    if study_num is not None:
        kwargs2 = {
            "x": 0.85,
            "y": 0.01,
            "s": rf"{study_num:<3d}",
            "color": "blue",
            "transform": ax.transAxes,
            "size": 14,
        }
        if study_num_kwargs is not None:
            kwargs2.update(study_num_kwargs)
        ax.text(**kwargs2)
    sublabel = chr(ord(sublabel) + 1)

    if ylabel:
        ax.set_ylabel("y (mm)", fontsize=labelfontsize)

    return cmesh, sublabel


def add_text_to_hist(
    props: AnalyzedProps, ax: Axes, dw: DictionaryWrapper, hist: Histogram, x: float = 0.6, y: float = 0.7, fontsize: int = 9, text: str = None,
):

    dwd = dw._DictionaryWrapper__dictionary
    h_fwhm = dwd['h_fwhm']
    if h_fwhm is None:
        h_fwhm = hist.hh[1] - hist.hh[0]
    v_fwhm = dwd['v_fwhm']
    if v_fwhm is None:
        v_fwhm = hist.vv[1] - hist.vv[0]

    if text is None:
        text = (
            rf"{'fH':<3} = {h_fwhm * 1000: 3.1f}" + "\n"
            rf"{'fV':<3} = {v_fwhm * 1000: 3.1f}" + "\n"
            rf"{'pH':<3} = {dwd['h_peak']* 1000: 3.1f}" + "\n"
            rf"{'pV':<3} = {dwd['v_peak']* 1000: 3.1f}" + "\n"
            rf"{'pI':<3} = {dwd['peak_intensity']:{props.photon_count_str}}"
        )
        text = text.replace("e+", "e")
    ax.text(
        x,
        y,
        text,
        color="black",
        alpha=0.9,
        fontsize=fontsize,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7),
        transform=ax.transAxes,
    )
    return text


def plot_pareto_2d(
    props: AnalyzedProps, ax, xlabel, ylabel, sublabel: str, ground=None, fontsize=14, annotate=False, legend=False, legend_loc=None
):
    x = np.array([t.values[PARETO_PLOT_ORDER[xlabel]] for t in props.study.trials])
    y = np.array([t.values[PARETO_PLOT_ORDER[ylabel]] for t in props.study.trials])

    norm = mpl.colors.Normalize(0, props.ntrials)
    cscatter = ax.scatter(
        x[props.other], y[props.other], c=props.other, cmap=cmm.tropical_r, marker="o", alpha=0.5, norm=norm
    )
    ax.scatter(
        x[props.pf], y[props.pf], c=props.pf, label="PF", cmap=cmm.tropical_r, marker=r"$\odot$", s=175, norm=norm
    )
    ax.scatter(
        x[props.ns], y[props.ns], c=props.ns, label="NS", cmap=cmm.tropical_r, marker=r"$\oplus$", s=175, norm=norm
    )

    if ground is not None:
        ax.scatter(ground[xlabel], ground[ylabel], color="red", marker="*", s=175, label="M")

    ax.set_xscale("log")
    if ylabel not in ["NLPI", "LWSI"]:
        ax.set_yscale("log")
        ylabel = rf"{ylabel} (mm)"

    ax.set_xlabel(rf"{xlabel} (mm)", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if annotate:
        for t in props.study.best_trials:
            ax.annotate(t.number, xy=(x[t.number], y[t.number]), xytext=(3, 3), textcoords="offset points")


    ax.text(0.02, 0.92, f"{sublabel})", transform=ax.transAxes, size=16, color="red", fontweight="bold")
    if legend:
        if legend_loc is None:
            ax.legend(loc = "best", framealpha=0.8)
        else:
            ax.legend(loc=legend_loc, framealpha=0.8)

    sublabel = chr(ord(sublabel) + 1)
    return cscatter, sublabel
