import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import os
import numpy as np


def plot_data(data, x_axis, value, is_test=False, condition=None, smooth=1, **kwargs):

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
        episodes = range(data.shape[0])
        if is_test:
            episodes = range(5, (data.shape[0]+1)*5, 5)
            print(data.shape[0])
            print(len(list(episodes)))
        data = pd.DataFrame({'Rewards': data, 'Episodes': episodes})

    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=x_axis, y=value, hue=condition, err_style="band", ci=95,**kwargs)
    plt.legend(loc="best").set_draggable(True)

    xscale = np.max(np.asarray(data[x_axis])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))



    plt.tight_layout(pad=0.5)

def plot_info_finals_state(data, x_axis, value, only_goal=True, is_test=False, condition=None, smooth=0, **kwargs):

    if only_goal:
        data = (data[:, 1] == "Goal").astype(int)
    else:
        data = np.isin(data[:, 1], ["Goal", "Cross Finish Line"]).astype(int)

    if smooth == 0:
        success_rate = np.array([data[:t].sum() / t for t in range(1, data.shape[0] + 1)])
    else:

        y = np.ones(smooth)
        z = np.ones(data.shape[0])
        success_rate = np.convolve(data, y, 'same') / np.convolve(z, y, 'same')
        # success_rate = np.array([data[t-window:t].sum() / window for t in range(1, data.shape[0] + 1)])

    episodes = range(success_rate.shape[0])
    if is_test:
        episodes = range(5, (success_rate.shape[0]+1)*5, 5)
    data = pd.DataFrame({'Success Rate': success_rate, 'Episodes': episodes})

    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=x_axis, y=value, hue=condition, err_style="band", ci=95,**kwargs)
    plt.legend(loc="best").set_draggable(True)

    xscale = np.max(np.asarray(data[x_axis])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

def plot_extra_info(data, exo_data, finals_states=None):

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes = axes.flatten()
    idxs = [[0, 1], 3]
    for episode, ed in enumerate(exo_data):
        exo_veh_pos = [exo_veh[:, :2] for exo_veh in ed] # = len(exo_veh)
        exo_veh_extent = [exo_veh[0, -7:-5] for exo_veh in ed] # is not change
        for exo_pos, exo_vol in zip(exo_veh_pos, exo_veh_extent):
            x = exo_pos[0, 0] - exo_vol[0]
            y = exo_pos[0, 1] - exo_vol[1]
            rect = Rectangle((x, y), exo_vol[0]*2, exo_vol[1]*2, linewidth=2, edgecolor="r", facecolor='blue')
            axes[0].add_patch(rect)
        break

    if finals_states is not None:
        assert finals_states.shape[0] == data.shape[0]

    for episode, d in enumerate(data):
        color = (episode / len(data), 0, 0)
        if finals_states is not None:
            if finals_states[episode, 1] in ["Goal", "Cross Finish Line"]:
                color = (0, 0, episode / len(data))
            elif  finals_states[episode, 1] == "Collision Other":
                color = (0, episode / len(data), 0)
        axes[0].plot(d[:, 0], d[:, 1], color=color) # pos
        axes[1].plot(d[:, 3], color=color)

    axes[0].set_xlim([100, 220])
    axes[0].set_ylim([40, 80])
    plt.show()

def plot_losses(data, method):

    have_IL = method in ["CoL", "TD3CoL"]
    n_losses = 3 if have_IL else 2
    fig, axes = plt.subplots(1, n_losses, figsize=(16, 6))
    axes = axes.flatten()
    data = np.array(data)
    if have_IL:
        axes[0].plot(data[:, 0], label="BC Loss")
        axes[1].plot(data[:, 1], label="Actor Q Loss")
        axes[2].plot(data[:, 2], label="1-step Q learning Loss")
    else:
        axes[0].plot(data[:, 0], label="Actor Loss")
        axes[1].plot(data[:, 1], label="Critic Loss")

    plt.legend(loc="best").set_draggable(True)
    plt.show()

if __name__ == "__main__":

    import pathlib
    prefix_path  = os.path.join(pathlib.Path(__file__).parent.resolve().parent, "models_logs")
    vision = "VAE"
    learn_alg = "CoL"
    alg = os.path.join(vision, learn_alg)
    date = "2021-10-19-11-54"
    path = os.path.join(prefix_path, alg, date)

    info_final_states = np.load(os.path.join(path, "info_finals_state.npy"))
    test_info_final_states = np.load(os.path.join(path, "test_info_finals_state.npy"))

    rewards = np.load(os.path.join(path, "rewards.npy"))
    # rewards_wout = np.load(os.path.join(path, "rewards_wout.npy"))
    test_rewards = np.load(os.path.join(path, "test_rewards.npy"))
    #plot_data(rewards, "Episodes", "Rewards", smooth=4, label="w/ mov_avg", markers=True, dashes=False)
    plot_data(rewards, "Episodes", "Rewards", smooth=20, label="Train Reward",
              markers=True, dashes=False)
    plot_data(test_rewards, "Episodes", "Rewards", smooth=20, label="Test Reward",
              is_test=True, markers=True, dashes=False)
    #plot_data(rewards_wout, "Episodes", "Rewards", smooth=5, label="Train Reward Wout",
    #          markers=True, dashes=False)
    plt.show()

    plot_info_finals_state(info_final_states, "Episodes", "Success Rate", label="Train Success Rate", smooth=20, only_goal=False)
    plot_info_finals_state(test_info_final_states, "Episodes", "Success Rate", is_test=True, label="Test Success Rate", smooth=20, only_goal=False)
    plt.show()

    data = np.load(os.path.join(path, "train_agent_extra_info.npy"), allow_pickle=True)
    exo_data = np.load(os.path.join(path, "train_exo_agents_extra_info.npy"), allow_pickle=True)
    plot_extra_info(data, exo_data, info_final_states)

    data = np.load(os.path.join(path, "test_agent_extra_info.npy"), allow_pickle=True)
    exo_data = np.load(os.path.join(path, "test_exo_agents_extra_info.npy"), allow_pickle=True)
    plot_extra_info(data, exo_data, test_info_final_states)

    losses = np.load(os.path.join(path, "train_historical_losses.npy"), allow_pickle=True)
    plot_losses(losses, learn_alg)
