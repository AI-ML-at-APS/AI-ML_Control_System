import optuna
import joblib
import matplotlib.pyplot as plt
import os
import aps
from pathlib import Path


def setup_work_dir():
    root_dir = Path(aps.__path__[0]).parents[0]
    work_dir = root_dir / "work_directory/28-ID"
    os.chdir(work_dir)


if __name__ == "__main__":
    setup_work_dir()
    autofocus_trials = joblib.load("autofocus_final_20_2022:11:16:16:57.gz")

    # Example of a "trial"
    print(autofocus_trials[0])

    # Printing the beam info for the trial
    print(autofocus_trials[0].user_attrs["dw"])

    # For multiobjective optimization
    study = optuna.create_study(directions=["minimize", "minimize"])

    # For single objective
    # study = optuna.create_study(directions=["minimize"])

    # Load trials onto the study
    study.add_trials(autofocus_trials)

    # Generating the pareto front for the multiobjective optimization
    optuna.visualization.matplotlib.plot_pareto_front(study, target_names=["centroid", "sigma"])
    plt.tight_layout()

    # plt.savefig(...) to save the image
    plt.show()

    # Plotting the optimization histories
    # centroid first
    optuna.visualization.matplotlib.plot_optimization_history(
        study, target=lambda t: t.values[0], target_name="centroid"
    )
    plt.tight_layout()
    plt.show()

    # sigma
    optuna.visualization.matplotlib.plot_optimization_history(study, target=lambda t: t.values[1], target_name="sigma")
    plt.tight_layout()
    plt.show()

    # Other visualization options here
    # https://optuna.readthedocs.io/en/stable/reference/visualization/matplotlib.html

    # convert to pandas dataframe
    df = study.trials_dataframe()
    print(df)
