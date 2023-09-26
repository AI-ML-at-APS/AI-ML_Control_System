#%%
import configparser
import os
from datetime import datetime
from pathlib import Path

import imageio
import joblib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import optuna
import scipy

import aps
import aps.ai.autoalignment.beamline28IDB.optimization.analysis_plot_utils as apu
import aps.ai.autoalignment.beamline28IDB.optimization.analysis_utils as analysis


# %%
def get_progressive_trials(
    trials, directions
):
    progress_trials = [trials[0]]
    for i, t in enumerate(trials):
        if i == 0:
            continue
        
        dominated = False
        for other in trials[:i]:
            dominated = analysis._dominates(other, t, directions)
            if dominated:
                break
         
        if not dominated:
            progress_trials.append(t)
    return progress_trials
#%%
exp_type = "peak_fwhm_nlpi"
# base_dir = Path(aps.__file__).parents[1]
# autoalign_dir = base_dir / f'work_directory/28-ID/AI/autoalignment/'
autoalign_dir = Path("/Users/skandel/Library/CloudStorage/Box-Box/Beamtime_28ID_Nov2022/AI/autofocusing/")
data_base_dir = autoalign_dir / exp_type / "all_motors_worse"
print(f"data base dir exists {data_base_dir.exists()}")

#%%
data_base_dir.exists()

#%%
#log_file = Path(data_base_dir / "peak_fwhm_nlpi_moo_optimization_final_101_2022-11-21_14:32.gz")
log_file = Path(data_base_dir / "peak_fwhm_nlpi_moo_optimization_final_101_2022-11-22_10:19.gz")
log_file.exists()

#%%
hist_dir = Path(data_base_dir / "peak_fwhm_nlpi_moo_100_2022-11-22_steps")
hist_dir.exists()

#%%
study = analysis.create_study_from_trials(log_file, n_objectives=3)
#%%
n_steps = len(study.trials)
hists = analysis.load_histograms_from_files(n_steps, hist_dir, extension="gz")


#%%

#%%

#%%
nash_equil, trial_ix, n_dominated = analysis.select_nash_equil_trial_from_pareto_front(study)
print(trial_ix, n_dominated, nash_equil.number)
plt.imshow(hists[n_dominated[trial_ix]].data_2D)
plt.show()
#%%
# Getting the number from previous analysis
prog_trials = get_progressive_trials(study.trials[:nash_equil.number + 1], study.directions)
prog_trials


#%%

open_figs = len(plt.get_fignums())
for _ in range(open_figs):
    plt.close()

for t in study.best_trials:
    i = t.number
    plt.figure()
    plt.pcolormesh(hists[i].hh[700:-700], hists[i].vv[700:-700], hists[i].data_2D[700:-700, 700:-700].T)
    plt.title(i)
    plt.show()

#%%

open_figs = len(plt.get_fignums())
for _ in range(open_figs):
    plt.close()


ims = []
# plt.title("Initial structure")
fig = plt.figure()

for trial in prog_trials:
    i = trial.number

    time = datetime.strftime(trial.datetime_start, "%Y/%m/%d::%H:%M:%S")
    
    im = plt.pcolormesh(
        hists[i].hh[700:-700], hists[i].vv[700:-700], hists[i].data_2D[700:-700, 700:-700].T, animated=True, cmap=apu.CMAP)
    imt = plt.text(0.7, 0.95, time, transform=plt.gca().transAxes, color="black")
    imt2 = plt.text(0.05, 0.95, f"Trial {i}", transform=plt.gca().transAxes, color="black")
    ims.append([im, imt, imt2])
plt.xlabel(r"H $(mm)$")
plt.ylabel(r"V $(mm)$")

plt.axhline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
plt.axvline(0, color="gray", ls="--", linewidth=1, alpha=0.7)
plt.tight_layout()
ani = animation.ArtistAnimation(fig, ims, interval=350, repeat=False, blit=True)
ani.save("video.mp4")
plt.show()

#%%