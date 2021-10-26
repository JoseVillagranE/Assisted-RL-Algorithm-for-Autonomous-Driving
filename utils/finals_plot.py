import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
plt.style.use("ggplot")

def final_plots(data, n_situations, smooth=20):
    color = ["red", "green", "blue"]
    i = 0
    try:
        for alg, _data in data.items():
            for j, d in enumerate(_data):
                y = np.ones(smooth)
                z = np.ones(d.shape[0])
                d = np.convolve(d, y, 'same') / np.convolve(z, y, 'same')
                _data[j] = d

            _data = np.stack(_data, axis=0)
            mean = np.mean(_data, axis=0)
            std = np.std(_data, axis=0)
            episodes = range(5, (mean.shape[0]+1)*5, 5)
            plt.plot(episodes, mean, "-")
            plt.fill_between(episodes, mean + std, mean - std, alpha=0.2, label=alg, color=color[i])
            i += 1

        plt.xlabel("Episodes", fontsize=12, fontweight="bold")
        plt.ylabel("Reward", fontsize=12, fontweight="bold")
        plt.legend()
        plt.show()
    except ValueError:
        print(f"algo: {alg}")
        for d in _data: print(d.shape)

    # fig, axes = plt.subplots(2, n_situations, figsize=(20, 10))
    # axes = axes.flatten()
    # for i, ax in enumrate(axes):
    #
    #     ax.plot()

def get_data(date, learn_alg, stat="reward", mode="train"):
    prefix_path  = os.path.join(pathlib.Path(__file__).parent.resolve().parent, "models_logs")
    vision = "VAE"
    alg = os.path.join(vision, learn_alg)
    path = os.path.join(prefix_path, alg, date)
    if stat == "reward":
        if mode == "train":
            file = "rewards.npy"
        else:
            file = "test_rewards.npy"
    elif stat == "success":
        if mode == "reward":
            file = "train_agent_extra_info.npy"
        else:
            file = "test_agent_extra_info.npy"
    data = np.load(os.path.join(path, file), allow_pickle=True)
    return data

if __name__ == "__main__":

    test_reward_data = {
                "DDPG": [get_data("2021-09-18-05-09", "DDPG", "reward", "test")[:139],
                get_data("2021-09-21-05-15", "DDPG", "reward", "test")[:139],
                # get_data("2021-09-21-16-46", "DDPG", "reward", "test"),
                get_data("2021-09-21-16-59", "DDPG", "reward", "test")
                ],
                "CoL": [get_data("2021-09-22-15-50", "CoL", "reward", "test")
                ],
                "TD3CoL": [get_data("2021-10-22-13-54", "TD3CoL", "reward", "test"),
                get_data("2021-10-22-16-07", "TD3CoL", "reward", "test"),
                get_data("2021-10-22-18-22", "TD3CoL", "reward", "test")
                ]

            }

    final_plots(test_reward_data, n_situations=1)
