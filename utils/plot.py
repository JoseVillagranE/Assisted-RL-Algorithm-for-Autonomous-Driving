import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np


def plot_data(data, x_axis, value, condition=None, smooth=1, **kwargs):

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
        data = pd.DataFrame({'Rewards': data, 'Episodes': range(data.shape[0])})

    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=x_axis, y=value, hue=condition, err_style="band", ci=95,**kwargs)
    plt.legend(loc="best").set_draggable(True)

    xscale = np.max(np.asarray(data[x_axis])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)


if __name__ == "__main__":

    # data = sns.load_dataset("flights")
    # print(data.head())
    #
    # may_fl = data.query("month == 'May'")
    # sns.lineplot(data=data, x="year", y="passengers")

    rewards = np.load("rewards.npy")
    avg_rewards = np.load("avg_rewards.npy")
    plot_data(rewards, "Episodes", "Rewards", smooth=4, label="w/ mov_avg", markers=True, dashes=False)
    plot_data(rewards, "Episodes", "Rewards", smooth=1, label="w/out mov_avg", markers=True, dashes=False)
    plot_data(avg_rewards, "Episodes", "Rewards", smooth=1, label="avg", markers=True, dashes=False)
    plt.show()

    print(f"kl\t"
            f"zacowea")
