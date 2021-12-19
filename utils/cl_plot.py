import seaborn as sns
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import ListedColormap
import pathlib
import pandas as pd
from scipy.interpolate import interp1d
import copy
# plt.style.use("ggplot")

COLOR = {"DDPG": "red", "CoL": "green", "TD3CoL": "blue"}

import matplotlib.legend as mlegend

BASE_TRAINING_EPISODES = 6
TRAINING_EPISODES = 5


def get_cmap(n, name='Paired'):
    return plt.cm.get_cmap(name, n)

def plot_cl_reward(rewards, gt_tasks_indexs):

    n_tasks = len(rewards)
    print(f"n_tasks: {n_tasks}")
    tasks_indexes = [0]*n_tasks
    cmap = get_cmap(n_tasks)

    fig, ax = plt.subplots(figsize=(16, 6))
    # sns.set(style="darkgrid", font_scale=1.5)
    base_training_reward = np.array(rewards[0][:BASE_TRAINING_EPISODES])
    plot_data(base_training_reward,
            np.arange(BASE_TRAINING_EPISODES),
            "Episodes",
            "Rewards",
            ax=ax,
            smooth=4,
            color=cmap(0),
            markers=True,
            dashes=False)
    ar1 = plt.axvspan(0, BASE_TRAINING_EPISODES, facecolor='red', alpha=0.15, linestyle = None)
    ar2 = plt.axvspan(BASE_TRAINING_EPISODES, BASE_TRAINING_EPISODES + n_tasks*TRAINING_EPISODES, facecolor='green', alpha=0.15, linestyle = 'None')
    ar3 = plt.axvspan(BASE_TRAINING_EPISODES + n_tasks*TRAINING_EPISODES, BASE_TRAINING_EPISODES + (n_tasks+gt_tasks_indexs.shape[0])*TRAINING_EPISODES, facecolor='blue', alpha=0.15, linestyle = 'None')
    tasks_indexes[0] += BASE_TRAINING_EPISODES

    training_indexes = list(range(n_tasks))
    tasks_no_plotted = training_indexes.copy()
    training_indexes.extend(gt_tasks_indexs.tolist())
    for j, i in enumerate(training_indexes):
        data = np.array(rewards[i][tasks_indexes[i]:tasks_indexes[i]+TRAINING_EPISODES])
        label = "task: " + str(i) if i in tasks_no_plotted else None
        plot_data(data,
                np.arange(BASE_TRAINING_EPISODES+j*TRAINING_EPISODES, BASE_TRAINING_EPISODES+(j+1)*TRAINING_EPISODES),
                "Episodes",
                "Rewards",
                ax=ax,
                smooth=4,
                color=cmap(i),
                label=label)
        tasks_indexes[i] += TRAINING_EPISODES
        if i in tasks_no_plotted:
            tasks_no_plotted.remove(i)

    ax.annotate("Base Training", xy=(0.05, 1.0), xycoords="axes fraction", fontsize=12, fontweight="bold")
    ax.annotate("Categorizing Tasks", xy=(0.2, 1.0), xycoords="axes fraction", fontsize=12, fontweight="bold")
    ax.annotate("General Training", xy=(0.65, 1.0), xycoords="axes fraction", fontsize=12, fontweight="bold")
    legend1 = plt.legend(loc="best")#.set_draggable(True)
    plt.show()


def plot_data(data, episodes, x_axis, value, ax, is_test=False, condition=None, smooth=1, **kwargs):

    if smooth > 1:
        """
        smooth dataa w/ moving window avg
        smoothed_y[t] = avg(y[t-k], y[t-k+1], ..., y[t+k])
        """
        y = np.ones(smooth)
        z = np.ones(data.shape[0])
        smoothed_x = np.convolve(data, y, 'same') / np.convolve(z, y, 'same')
        data = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    elif isinstance(data, np.ndarray):
        data = pd.DataFrame({'Rewards': data, 'Episodes': episodes})

    sns.lineplot(data=data, x=x_axis, y=value, ax=ax, hue=condition, err_style="band", ci=95,**kwargs)

if __name__ == "__main__":

    import pathlib
    prefix_path  = os.path.join(pathlib.Path(__file__).parent.resolve().parent, "models_logs")
    vision = "VAE"
    learn_alg = "TD3CoL"
    alg = os.path.join("CL", vision, learn_alg)
    date = "2021-12-10-18-20"
    path = os.path.join(prefix_path, alg, date)

    info_final_states = np.load(os.path.join(path, "info_finals_state.npy"))
    rewards = np.load(os.path.join(path, "rewards.npy"), allow_pickle=True)
    I = np.load(os.path.join(path, "I_stats.npy"))

    # curriculum learning reward plot
    plot_cl_reward(rewards, I)
