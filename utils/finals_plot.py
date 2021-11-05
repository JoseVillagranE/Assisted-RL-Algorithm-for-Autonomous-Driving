import seaborn as sns
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
plt.style.use("ggplot")

COLOR = {"DDPG": "red", "CoL": "green", "TD3CoL": "blue"}

def best_choosing(list_of_data, only_goal, smooth=20):
    for i, data in enumerate(list_of_data):
        for alg, _data in data.items():
            for j, d in enumerate(_data):
                if only_goal:
                    d = (d[:, 1] == "Goal").astype(int)
                else:
                    d = np.isin(d[:, 1], ["Goal", "Cross Finish Line"]).astype(int)

                if smooth == 0:
                    success_rate = np.array([d[:t].sum() / t for t in range(1, data.shape[0] + 1)])
                else:
                    y = np.ones(smooth)
                    z = np.ones(d.shape[0])
                    success_rate = np.convolve(d, y, 'same') / np.convolve(z, y, 'same')
                _data[j] = success_rate
            for k, d in enumerate(_data):
                episodes = range(5, (d.shape[0]+1)*5, 5)
                plt.plot(episodes, d, label=k)
            plt.title(f"alg:{alg} || i:{i}")
            plt.legend(loc="best")
            plt.show()


def test_compound_agg():

    n_samples = [100, 50, 50]
    data = [np.random.normal(size=n_samples[i]) for i in range(3)]
    mean, std = compound_agg(data)
    episodes = range(max(n_samples))
    plt.plot(episodes, mean, "-", color="blue")
    plt.fill_between(episodes, mean + std, mean - std, alpha=0.1, color="red")
    plt.show()


def compound_agg(data):
    shapes = [d.shape[0] for d in data]
    max_shapes = max(shapes)
    min_shapes = min(shapes)
    mean = data[shapes.index(max_shapes)]
    std = np.zeros(max(shapes))
    t0 = 0
    t1 = min_shapes
    for i in range(len(set(shapes))):
        current_data = [d[t0:t1] for d in data]
        stacked_data = np.stack(current_data, axis=0)
        mean[t0:t1] = np.mean(stacked_data, axis=0)
        std[t0:t1] = np.std(stacked_data, axis=0)
        for i in reversed(range(len(data))):
            if len(data[i]) == t1:
                del data[i]
                del shapes[i]
        if len(data) in [0, 1]:
            break
        t0 = t1
        t1 = min(shapes)

    return mean, std

def test_collision_plots(list_of_data, labels):

    prefix_path  = os.path.join(pathlib.Path(__file__).parent.resolve().parent, "models_logs", "VAE", "Eval")
    bar_width = 0.1
    delta = {"DDPG": -bar_width, "CoL": 0, "TD3CoL": bar_width}
    colors = {"DDPG": ("red", "pink"),
                "CoL": ("green", "lightgreen"),
                "TD3CoL": ("blue", "paleturquoise", "aqua")}

    for i, data in enumerate(list_of_data):
        for alg, _data in data.items():
            for j, d in enumerate(_data):
                d = np.load(os.path.join(prefix_path, alg, d, "finish_reasons.npy"), allow_pickle=True)
                print(d)
                if j == 0:
                    full_data = d
                else:
                    np.append(full_data, d)
            full_data_length = full_data.shape[0]
            print(f"n_situation: {i} || alg: {alg} || full length data: {full_data_length}")
            collision_veh_ratio = full_data[full_data == "Collision Veh"].shape[0] /  full_data_length
            collision_other_ratio = full_data[full_data == "Collision Other"].shape[0] /  full_data_length
            goal_ratio = full_data[full_data == "Goal"].shape[0] /  full_data_length
            plt.bar(i+delta[alg], goal_ratio, width=bar_width, color=colors[alg][0], bottom=0)
            plt.bar(i+delta[alg], collision_veh_ratio, width=bar_width, color=colors[alg][1], bottom=goal_ratio)
            plt.bar(i+delta[alg], collision_other_ratio, width=bar_width, color=colors[alg][2], bottom=collision_veh_ratio)

    plt.xticks(list(range(len(labels))), labels)
    plt.show()


def success_plots(list_of_data, n_situations, n_exo_agents, title, only_goal=False, smooth=20, add_compound_agg=False):

    fig, axes = plt.subplots(1 + ((n_situations-1) // 3), 3, figsize=(20, 8))
    axes = axes.flatten()
    xlims = [100, 400, 500, 1000, 1000, 1000]
    for i, (data, ax) in enumerate(zip(list_of_data, axes)):
        for alg, _data in data.items():
            for j, d in enumerate(_data):
                if only_goal:
                    d = (d[:, 1] == "Goal").astype(int)
                else:
                    d = np.isin(d[:, 1], ["Goal", "Cross Finish Line"]).astype(int)

                if smooth == 0:
                    success_rate = np.array([d[:t].sum() / t for t in range(1, data.shape[0] + 1)])
                else:
                    y = np.ones(smooth)
                    z = np.ones(d.shape[0])
                    success_rate = np.convolve(d, y, 'same') / np.convolve(z, y, 'same')
                _data[j] = success_rate
            if add_compound_agg:
                mean, std = compound_agg(_data)
            else:
                _data = np.stack(_data, axis=0)
                mean = np.mean(_data, axis=0)
                std = np.std(_data, axis=0)
            episodes = range(5, (mean.shape[0]+1)*5, 5)
            ax.plot(episodes, mean, "-", color=COLOR[alg], label=alg)
            ax.fill_between(episodes, mean + std, mean - std, alpha=0.1, color=COLOR[alg])

        ax.set_xlim((0, xlims[i]))
        ax.set_xlabel("Episodios", fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(f"N: {n_exo_agents[i]}", fontsize=16, fontweight="bold")
        ax.set_ylim((-0.05, 1.05))
    axes[0].legend(loc="lower left")
    plt.show()

def reward_plots(list_of_data, n_situations, n_exo_agents, title, smooth=20, add_compound_agg=False):

    try:
        fig, axes = plt.subplots(1 + ((n_situations-1) // 3), 3, figsize=(20, 8))
        axes = axes.flatten()
        xlims = [100, 400, 500, 1000, 1000, 1000]
        for i, (data, ax) in enumerate(zip(list_of_data, axes)):
            for alg, _data in data.items():
                for j, d in enumerate(_data):
                    y = np.ones(smooth)
                    z = np.ones(d.shape[0])
                    d = np.convolve(d, y, 'same') / np.convolve(z, y, 'same')
                    _data[j] = d

                if add_compound_agg:
                    mean, std = compound_agg(_data)
                else:
                    _data = np.stack(_data, axis=0)
                    mean = np.mean(_data, axis=0)
                    std = np.std(_data, axis=0)
                episodes = range(5, (mean.shape[0]+1)*5, 5)
                ax.plot(episodes, mean, "-", color=COLOR[alg], label=alg)
                ax.fill_between(episodes, mean + std, mean - std, alpha=0.1, color=COLOR[alg])
            ax.set_xlim((0, xlims[i]))
            ax.set_xlabel("Episodios", fontsize=12, fontweight="bold")
            ax.set_ylabel(title, fontsize=12, fontweight="bold")
            ax.set_title(f"N: {n_exo_agents[i]}", fontsize=16, fontweight="bold")
            ax.set_ylim((-1000, 1000))
        axes[0].legend(loc="lower left")
        plt.show()
    except ValueError as error:
        print(error)
        print(f"algo: {alg}")
        print(f"idx: {i}")
        for d in _data: print(d.shape)

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
        if mode == "train":
            file = "info_finals_state.npy"
        else:
            file = "test_info_finals_state.npy"
    data = np.load(os.path.join(path, file), allow_pickle=True)
    return data

if __name__ == "__main__":

    test_reward_data = [
        {
            "DDPG": [get_data("2021-09-15-14-51", "DDPG", "reward", "test"),
            get_data("2021-09-15-18-29", "DDPG", "reward", "test"),
            get_data("2021-09-15-20-17", "DDPG", "reward", "test")
            ],
            "CoL": [get_data("2021-10-19-11-54", "CoL", "reward", "test"),
            get_data("2021-10-19-12-28", "CoL", "reward", "test"),
            get_data("2021-10-19-12-59", "CoL", "reward", "test")
            ],
            "TD3CoL": [get_data("2021-10-23-20-46", "TD3CoL", "reward", "test"),
            get_data("2021-10-23-21-15", "TD3CoL", "reward", "test"),
            get_data("2021-10-23-21-40", "TD3CoL", "reward", "test")
            ]
        }, # 0
        {
            "DDPG": [get_data("2021-09-15-23-39", "DDPG", "reward", "test"),
            get_data("2021-09-16-02-12", "DDPG", "reward", "test"),
            get_data("2021-09-16-05-29", "DDPG", "reward", "test")
            ],
            "CoL": [get_data("2021-10-24-16-58", "CoL", "reward", "test"),
            get_data("2021-10-24-18-58", "CoL", "reward", "test"),
            get_data("2021-10-24-20-54", "CoL", "reward", "test")
            ],
            "TD3CoL": [get_data("2021-10-22-21-07", "TD3CoL", "reward", "test"),
            get_data("2021-10-22-23-15", "TD3CoL", "reward", "test"),
            get_data("2021-10-23-18-06", "TD3CoL", "reward", "test")
            ]
        }, # 1
        {
            "DDPG": [get_data("2021-09-18-05-09", "DDPG", "reward", "test"),
            get_data("2021-09-21-05-15", "DDPG", "reward", "test"),
            get_data("2021-09-21-16-59", "DDPG", "reward", "test")
            ],
            "CoL": [get_data("2021-10-23-22-42", "CoL", "reward", "test"),
            get_data("2021-10-24-10-09", "CoL", "reward", "test"),
            get_data("2021-10-24-12-56", "CoL", "reward", "test")
            ],
            "TD3CoL": [get_data("2021-10-30-23-18", "TD3CoL", "reward", "test"),
            # get_data("2021-10-31-11-12", "TD3CoL", "reward", "test"),
            get_data("2021-10-31-13-56", "TD3CoL", "reward", "test")
            ]
        }, # 2
        {
            "DDPG": [get_data("2021-10-24-23-02", "DDPG", "reward", "test"),
            get_data("2021-10-25-09-48", "DDPG", "reward", "test"),
            get_data("2021-10-25-14-52", "DDPG", "reward", "test")
            ],
            "CoL": [get_data("2021-11-02-15-03", "CoL", "reward", "test"),
            get_data("2021-11-04-10-22", "CoL", "reward", "test"),
            get_data("2021-10-03-19-27", "CoL", "reward", "test")
            ],
            "TD3CoL": [get_data("2021-10-31-16-07", "TD3CoL", "reward", "test"),
            get_data("2021-10-31-20-04", "TD3CoL", "reward", "test"),
            get_data("2021-10-22-11-37", "TD3CoL", "reward", "test")
            ]
        }, # 3
        {
            "DDPG": [get_data("2021-10-28-21-04", "DDPG", "reward", "test"),
            get_data("2021-10-29-11-35", "DDPG", "reward", "test"),
            get_data("2021-10-29-17-53", "DDPG", "reward", "test")
            ],
            "CoL": [get_data("2021-10-20-10-29", "CoL", "reward", "test"),
            get_data("2021-10-20-12-43", "CoL", "reward", "test"),
            get_data("2021-11-04-14-55", "CoL", "reward", "test")
            ],
            "TD3CoL": [get_data("2021-10-13-12-30", "TD3CoL", "reward", "test"),
            get_data("2021-11-01-15-30", "TD3CoL", "reward", "test"),
            get_data("2021-11-01-10-15", "TD3CoL", "reward", "test")
            ]

        }, # 3 con 1 en mov
        {
            "DDPG": [get_data("2021-10-30-10-11", "DDPG", "reward", "test")
            ],
            "CoL": [get_data("2021-11-01-19-55", "CoL", "reward", "test"),
            get_data("2021-11-02-10-15", "CoL", "reward", "test"),
            get_data("2021-10-19-20-20", "CoL", "reward", "test")
            ],
            "TD3CoL": [get_data("2021-10-21-09-33", "TD3CoL", "reward", "test"),
            get_data("2021-10-21-11-13", "TD3CoL", "reward", "test"),
            get_data("2021-10-21-14-23", "TD3CoL", "reward", "test")
            ]
        }, # 3 con 2 en mov
    ]

    test_success_data = [
        {
            "DDPG": [get_data("2021-09-15-14-51", "DDPG", "success", "test"),
            get_data("2021-09-15-18-29", "DDPG", "success", "test"),
            get_data("2021-09-15-20-17", "DDPG", "success", "test")
            ],
            "CoL": [get_data("2021-10-19-11-54", "CoL", "success", "test"),
            get_data("2021-10-19-12-28", "CoL", "success", "test"),
            get_data("2021-10-19-12-59", "CoL", "success", "test")
            ],
            "TD3CoL": [get_data("2021-10-23-20-46", "TD3CoL", "success", "test"),
            get_data("2021-10-23-21-15", "TD3CoL", "success", "test"),
            get_data("2021-10-23-21-40", "TD3CoL", "success", "test")
            ]
        }, # 0
        {
            "DDPG": [get_data("2021-09-15-23-39", "DDPG", "success", "test"),
            get_data("2021-09-16-02-12", "DDPG", "success", "test"),
            get_data("2021-09-16-05-29", "DDPG", "success", "test")
            ],
            "CoL": [get_data("2021-10-24-16-58", "CoL", "success", "test"),
            get_data("2021-10-24-18-58", "CoL", "success", "test"),
            get_data("2021-10-24-20-54", "CoL", "success", "test")
            ],
            "TD3CoL": [get_data("2021-10-22-21-07", "TD3CoL", "success", "test"),
            get_data("2021-10-22-23-15", "TD3CoL", "success", "test"),
            get_data("2021-10-23-18-06", "TD3CoL", "success", "test")
            ]
        }, # 1
        {
            "DDPG": [get_data("2021-09-18-05-09", "DDPG", "success", "test"),
            get_data("2021-09-21-05-15", "DDPG", "success", "test"),
            get_data("2021-09-21-16-59", "DDPG", "success", "test")
            ],
            "CoL": [get_data("2021-10-23-22-42", "CoL", "success", "test"),
            get_data("2021-10-24-10-09", "CoL", "success", "test"),
            get_data("2021-10-24-12-56", "CoL", "success", "test")
            ],
            "TD3CoL": [get_data("2021-10-30-23-18", "TD3CoL", "success", "test"),
            # get_data("2021-10-31-11-12", "TD3CoL", "success", "test"),
            get_data("2021-10-31-13-56", "TD3CoL", "success", "test")
            ]
        }, # 2
        {
            "DDPG": [get_data("2021-10-24-23-02", "DDPG", "success", "test"),
            get_data("2021-10-25-09-48", "DDPG", "success", "test"),
            get_data("2021-10-25-14-52", "DDPG", "success", "test")
            ],
            "CoL": [get_data("2021-11-02-15-03", "CoL", "success", "test"),
            get_data("2021-11-04-10-22", "CoL", "success", "test"),
            get_data("2021-10-03-19-27", "CoL", "success", "test")
            ],
            "TD3CoL": [get_data("2021-10-31-16-07", "TD3CoL", "success", "test"),
            get_data("2021-10-31-20-04", "TD3CoL", "success", "test"),
            get_data("2021-10-22-11-37", "TD3CoL", "success", "test")
            ]
        }, # 3
        {
            "DDPG": [get_data("2021-10-28-21-04", "DDPG", "success", "test"),
            get_data("2021-10-29-11-35", "DDPG", "success", "test"),
            get_data("2021-10-29-17-53", "DDPG", "success", "test")
            ],
            "CoL": [get_data("2021-10-20-10-29", "CoL", "success", "test"),
            get_data("2021-10-20-12-43", "CoL", "success", "test"),
            get_data("2021-11-04-14-55", "CoL", "success", "test")
            ],
            "TD3CoL": [get_data("2021-10-13-12-30", "TD3CoL", "success", "test"),
            get_data("2021-11-01-15-30", "TD3CoL", "success", "test"),
            get_data("2021-11-01-10-15", "TD3CoL", "success", "test")
            ]
        }, # 3 con 1 en mov
        {
            "DDPG": [get_data("2021-10-30-10-11", "DDPG", "success", "test")
            ],
            "CoL": [get_data("2021-11-01-19-55", "CoL", "success", "test"),
            get_data("2021-11-02-10-15", "CoL", "success", "test"),
            get_data("2021-10-19-20-20", "CoL", "success", "test")
            ],
            "TD3CoL": [get_data("2021-10-21-09-33", "TD3CoL", "success", "test"),
            get_data("2021-10-21-11-13", "TD3CoL", "success", "test"),
            get_data("2021-10-21-14-23", "TD3CoL", "success", "test")
            ]
        }, # 3 con 2 en mov
    ]

    eval_data = [

        {
            "TD3CoL": ["2021-11-04-08-46", "2021-11-04-08-58"]
        }

    ]


    # n_exo_agents = [0, 1, 2, 3, 31, 32]
    #
    # reward_plots(test_reward_data,
    #             n_exo_agents = n_exo_agents,
    #             n_situations=len(n_exo_agents),
    #             title="Recompensa Promedid",
    #             add_compound_agg=True)
    # success_plots(test_success_data,
    #                 n_exo_agents = n_exo_agents,
    #                 n_situations=len(n_exo_agents),
    #                 title="Radio de exito",
    #                 add_compound_agg=True)
    # best_choosing(test_success_data, only_goal=False)



    test_collision_plots(eval_data, ["test"])
