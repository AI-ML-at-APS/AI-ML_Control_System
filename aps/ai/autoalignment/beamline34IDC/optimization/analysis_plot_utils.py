import numpy as np
import cmasher as cmm
import matplotlib as mpl
from matplotlib.axes import Axes
from aps.ai.autoalignment.common.util.common import Histogram, DictionaryWrapper
import dataclasses as dt
from optuna import Study, Trial
from cycler import cycler


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
    float_format: str = "3.1f"
    photon_count_str: str = "3.1e"
    hist_nlpi_text: bool = True
    distance_units: str = 'mm'
    ntrials: int = dt.field(init=False)
    ns: list = dt.field(init=False)
    pf: list = dt.field(init=False)
    other: list = dt.field(init=False)
    unit_factor: float = dt.field(init=False)
    unit_str: str = dt.field(init=False)

    def __post_init__(self):
        self.ntrials = len(self.study.trials)
        self.ns = [self.nash_trial.number]
        self.pf = [t.number for t in self.study.best_trials if t.number != self.nash_trial.number]
        self.other = [n for n in range(len(self.study.trials)) if n not in self.ns and n not in self.pf]

        if self.distance_units == 'mm':
            self.unit_factor = 1
            self.unit_str = 'mm'
        elif self.distance_units == 'um':
            self.unit_factor = 1e3
            self.unit_str = r'$\mu$m'
        else:
            raise ValueError


def plot_hist_2d(
    props: AnalyzedProps,
    ax: Axes,
    hist: Histogram,
    labelfontsize: int = 12,
    ticks: list = None,
    sublabel: str = "A",
    sublabel_kwargs: dict = None,
    study_num: int = None,
    study_num_kwargs: dict = None,
    ylabel: bool = False,
    ticklabelsize=8,
):
    # norm = mpl.colors.LogNorm(props.min_count, props.max_count)
    norm = mpl.colors.Normalize(0, props.max_count)
    cmesh = ax.pcolormesh(hist.hh * props.unit_factor, hist.vv * props.unit_factor, hist.data_2D.T, cmap=CMAP, rasterized=True)# norm=norm, rasterized=True)

    ax.set_aspect("equal")

    lims = (-props.xylim * props.unit_factor, props.xylim * props.unit_factor)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.axhline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
    ax.axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)

    ax.tick_params(axis="both", which="major", labelsize=ticklabelsize)
    ax.tick_params(axis="both", which="minor", labelsize=ticklabelsize)
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    
    ax.set_xlabel(f"x ({props.unit_str})", fontsize=labelfontsize)

    kwargs1 = {
        "x": 0.02,
        "y": 0.9,
        "s": sublabel,
        "transform": ax.transAxes,
        "size": 14,
        "color": "red",
        "fontweight": "bold",
    }
    if sublabel_kwargs is not None:
        kwargs1.update(sublabel_kwargs)

    ax.text(**kwargs1)

    if study_num is not None:
        kwargs2 = {
            "x": 0.87,
            "y": 0.01,
            "s": rf"{study_num:<3d}",
            "color": "blue",
            "transform": ax.transAxes,
            "size": 12,
        }
        if study_num_kwargs is not None:
            kwargs2.update(study_num_kwargs)
        ax.text(**kwargs2)
    sublabel = chr(ord(sublabel) + 1)

    if ylabel:
        ax.set_ylabel(f"y ({props.unit_str})", fontsize=labelfontsize)

    return cmesh, sublabel


def add_text_to_hist(
    props: AnalyzedProps, ax: Axes, dw: DictionaryWrapper, x: float = 0.6, y: float = 0.7, fontsize: int = 9
):
    dwd = dw._DictionaryWrapper__dictionary


    text = (
        rf"{'fH':<3} = {dwd['h_fwhm'] * 1e3: {props.float_format}}" + "\n"
        rf"{'fV':<3} = {dwd['v_fwhm']* 1e3: {props.float_format}}" + "\n"
        rf"{'pH':<3} = {dwd['h_peak']* 1e3: {props.float_format}}" + "\n"
        rf"{'pV':<3} = {dwd['v_peak']* 1e3: {props.float_format}}"
    )
    if props.hist_nlpi_text:
        text = (text + "\n"
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


def get_trial_objective_values(study, val_index):
    y = []
    for t in study.trials:
        if t.values is None:
            y.append(np.nan)
        else:
            y.append(t.values[val_index])
    return np.array(y)
    

def plot_pareto_2d(
    props: AnalyzedProps, ax, xlabel, ylabel, sublabel: str, ground=None, fontsize=14, annotate=False, legend=False, ticklabelsize=10,
):
    x = get_trial_objective_values(props.study, PARETO_PLOT_ORDER[xlabel])
    y = get_trial_objective_values(props.study, PARETO_PLOT_ORDER[ylabel])
    #x = np.array([t.values[PARETO_PLOT_ORDER[xlabel]] for t in props.study.trials])
    #y = np.array([t.values[PARETO_PLOT_ORDER[ylabel]] for t in props.study.trials])

    norm = mpl.colors.Normalize(0, props.ntrials)

    if xlabel in ['FWHM', 'PL']:
        x = x * props.unit_factor
    if ylabel in ['FWHM', 'PL']:
        y = y * props.unit_factor

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
        if xlabel in ['FWHM', 'PL']:
            gx = ground [xlabel] * props.unit_factor
        if ylabel in ['FWHM', 'PL']:
            gy = ground[ylabel] * props.unit_factor
        ax.scatter(gx, gy, color="red", marker="*", s=175, label="M")

    ax.set_xscale("log")
    if ylabel != "NLPI":
        ax.set_yscale("log")
        ylabel = rf"{ylabel} ({props.unit_str})"

    ax.tick_params(axis="both", which="major", labelsize=ticklabelsize)
    ax.tick_params(axis="both", which="minor", labelsize=ticklabelsize)

    ax.set_xlabel(rf"{xlabel} ({props.unit_str})", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if annotate:
        for t in props.study.best_trials:
            ax.annotate(t.number, xy=(x[t.number], y[t.number]), xytext=(3, 3), textcoords="offset points")
    if legend:
        ax.legend(loc="best", framealpha=0.8)

    ax.text(0.02, 0.92, sublabel, transform=ax.transAxes, size=14, color="red", fontweight="bold")
    sublabel = chr(ord(sublabel) + 1)
    return cscatter, sublabel


def plot_1d_lines(props, ax, hist_test, hist_init=None, hist_ground=None, label=None, labelfontsize: int = 12, axis=0, cyc=None, sublabel=None, ticklabelsize=9):

    if cyc is None:
        cyc = cycler(ls='-')()

    ax.plot(hist_test.hh * props.unit_factor, 
            hist_test.data_2D.sum(axis=axis), label=label, **next(cyc), alpha=0.7, linewidth=2.0)
    if hist_init is not None:
        ax.plot(hist_init.hh * props.unit_factor,
                hist_init.data_2D.sum(axis=axis), label='I', **next(cyc), alpha=0.7, linewidth=2.0)
    if hist_ground is not None:
        ax.plot(hist_ground.hh * props.unit_factor,
                hist_ground.data_2D.sum(axis=axis), label='M', **next(cyc), alpha=0.7, linewidth=2.0)


    ax.tick_params(axis="both", which="major", labelsize=ticklabelsize)
    ax.tick_params(axis="both", which="minor", labelsize=ticklabelsize)

    lims = (-props.xylim * props.unit_factor, props.xylim * props.unit_factor)

    if axis == 0:
        ax.set_xlabel(f'x ({props.unit_str})', size=labelfontsize)  
    elif axis == 1:
        ax.set_xlabel(f'y ({props.unit_str})', size=labelfontsize)    
    ax.set_xlim(lims)  
    
    if sublabel is not None: 
        ax.text(0.02, 0.92, sublabel, transform=ax.transAxes, size=14, color="red", fontweight="bold")
        sublabel = chr(ord(sublabel) + 1)
    return sublabel

