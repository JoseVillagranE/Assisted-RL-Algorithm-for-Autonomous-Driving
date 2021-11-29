import seaborn as sns
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib
import pandas as pd
from scipy.interpolate import interp1d

plt.style.use("ggplot")

COLOR = {"DDPG": "red", "CoL": "green", "TD3CoL": "blue"}

import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle

def tablelegend(ax, col_labels=None, row_labels=None, title_label="", *args, **kwargs):
    """
    Place a table legend on the axes.

    Creates a legend where the labels are not directly placed with the artists,
    but are used as row and column headers, looking like this:

    title_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |


    Parameters
    ----------

    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.

    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.

    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.

    title_label : str, optional
        Label for the top left corner in the legend table.

    ncol : int
        Number of columns.


    Other Parameters
    ----------------

    Refer to `matplotlib.legend.Legend` for other parameters.

    """
    #################### same as `matplotlib.axes.Axes.legend` #####################
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    print(len(handles))
    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')

    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        title_label = [title_label]

        # blank rectangle handle
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]

        # empty label
        empty = [""]

        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol

        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow

        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_

def do_labeled_bar(p, ax, tol=0):
    labels = [round(v.get_height(), 2) if v.get_height() > tol else '' for v in p]
    ax.bar_label(p, labels=labels, label_type='center', fontweight='bold')

def circumference_samples(R, offset, n_samples, pi_limits):
    angle_samples = np.linspace(pi_limits[0], pi_limits[1], n_samples)
    xs = R*np.cos(angle_samples) + offset[0]
    ys = R*np.sin(angle_samples) + offset[1]
    return xs, ys

def plot_escenarios():

    x_limits = [98, 219]
    y_limits = [53, 68]
    break_points = [[152, 161], [176.89, 185.38]]

    prefix_path  = os.path.join(pathlib.Path(__file__).parent.resolve().parent, "models_logs", "VAE", "Eval")
    exo_data = np.load(os.path.join(prefix_path, "TD3CoL", "2021-11-16-16-06", "train_exo_agents_extra_info.npy"), allow_pickle=True)
    for episode, ed in enumerate(exo_data):
        exo_veh_pos_x = [exo_veh[:, 0] for exo_veh in ed]
        exo_veh_pos_y = [exo_veh[:, 1] for exo_veh in ed]
        plt.plot(exo_veh_pos_x, exo_veh_pos_y, 'ro')

    # upper trace
    plt.plot(x_limits, [y_limits[0]]*2, color="black")
    # 1st below trace
    plt.plot([x_limits[0], break_points[0][0]], [y_limits[1]]*2, color="black")
    # 2nd below trace
    plt.plot([break_points[1][1], x_limits[1]], [y_limits[1]]*2, color="black")

    # middle trace
    plt.plot([x_limits[0]+20, x_limits[1]], [61, 61], color="black")
    # below division trace
    plt.plot(x_limits, [64.5, 64.5], '--', color="black")
    # upper division trace
    plt.plot(x_limits, [57, 57], '--', color="black")
    # plot 1st curve
    xs, ys = circumference_samples(9, (152, 77), 100, (3*np.pi/2, 2*np.pi))
    plt.plot(xs, ys, color="black")
    xs, ys = circumference_samples(10.5, (185.38, 78.59), 100, (np.pi, 3*np.pi/2))
    plt.plot(xs, ys, color="black")
    plt.ylim((y_limits[1] + 5, y_limits[0] - 5))
    plt.show()


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


def rollup_agg(roll_ups, agg="mean"):
    for i, roll in enumerate(roll_ups):
        agg = roll.mean(axis=0) if agg=="mean" else roll.sum(axis=0)
        if i == 0:
            result = agg
        else:
            result = np.vstack((result, agg))
    return result

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
    bar_width = 0.2
    delta = {"DDPG": -bar_width, "CoL": 0, "TD3CoL": bar_width}
    my_cmap = ListedColormap(sns.color_palette())
    reds = sns.color_palette("Reds_r")
    greens = sns.color_palette("Greens_r")
    blues = sns.color_palette("Blues_r")
    purples = sns.cubehelix_palette(10)

    colors = {"DDPG": (reds[2], reds[1], reds[0]),
                "CoL": (blues[2], blues[1], blues[0]),
                "TD3CoL": (purples[3], purples[5], purples[7])}

    fig, ax = plt.subplots()
    row_labels = ["Goal", "Other Col", "Exo-veh col"]

    for i, data in enumerate(list_of_data):
        for alg, _data in data.items():
            for j, d in enumerate(_data):
                d = np.load(os.path.join(prefix_path, alg, d, "finish_reasons.npy"), allow_pickle=True)
                if j == 0:
                    full_data = d
                else:
                    full_data = np.append(full_data, d)
            full_data_length = full_data.shape[0]
            print(f"n_situation: {i} || alg: {alg} || full length data: {full_data_length}")
            collision_veh_ratio = full_data[full_data == "Collision Veh"].shape[0] /  full_data_length
            collision_other_ratio = full_data[full_data == "Collision Other"].shape[0] /  full_data_length
            goal_ratio = (full_data[full_data == "Goal"].shape[0] + full_data[full_data == "Cross Finish Line"].shape[0]) /  full_data_length
            if i == 0:
                p1 = ax.bar(i+delta[alg], goal_ratio, width=bar_width, color=colors[alg][0], bottom=0, label=row_labels[0]+", "+alg)
                p2 = ax.bar(i+delta[alg], collision_veh_ratio, width=bar_width, color=colors[alg][1], bottom=goal_ratio, label=row_labels[1]+", "+alg)
                p3 = ax.bar(i+delta[alg], collision_other_ratio, width=bar_width, color=colors[alg][2], bottom=collision_veh_ratio+goal_ratio, label=row_labels[2]+", "+alg)
            else:
                p1 = ax.bar(i+delta[alg], goal_ratio, width=bar_width, color=colors[alg][0], bottom=0)
                p2 = ax.bar(i+delta[alg], collision_veh_ratio, width=bar_width, color=colors[alg][1], bottom=goal_ratio)
                p3 = ax.bar(i+delta[alg], collision_other_ratio, width=bar_width, color=colors[alg][2], bottom=collision_veh_ratio+goal_ratio)
            do_labeled_bar(p1, ax, tol=0.05)
            do_labeled_bar(p2, ax, tol=0.05)
            do_labeled_bar(p3, ax, tol=0.05)

    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_title("Evaluación de la politica de conducción para distintas configuraciones de exo-agentes", fontsize=12, fontweight="bold")
    tablelegend(ax, col_labels=["DDPG", "CoL", "TD3CoL"], row_labels=row_labels, bbox_to_anchor=(0.05, 1.1), ncol=3)
    plt.show()


def stats_plots(list_of_data, labels):

    prefix_path  = os.path.join(pathlib.Path(__file__).parent.resolve().parent, "models_logs", "VAE", "Eval")
    colors = {"DDPG": ("red", "lightcoral", "pink"),
                "CoL": ("green", "lawngreen", "lightgreen"),
                "TD3CoL": ("blue", "paleturquoise", "aqua")}

    stats_per_situation = []
    for i, data in enumerate(list_of_data):
        stats_per_alg = {}
        for alg, _data in data.items():
            for j, d in enumerate(_data):
                d = np.load(os.path.join(prefix_path, alg, d, "rewards.npy"), allow_pickle=True)
                if j == 0:
                    full_data = rollup_agg(d, agg="sum")
                else:
                    full_data = np.vstack((full_data, rollup_agg(d, agg="sum")))
            stats_per_alg[alg] = full_data
        stats_per_situation.append(stats_per_alg)

    situations = []
    DDPG_rollouts = []
    CoL_rollouts = []
    TD3CoL_rollouts = []
    for i in range(len(stats_per_situation)):
        situations += [labels[i]]*stats_per_situation[i]["DDPG"].shape[0]
        if i == 0:
            DDPG_rollouts = stats_per_situation[i]["DDPG"]
            CoL_rollouts = stats_per_situation[i]["CoL"]
            TD3CoL_rollouts = stats_per_situation[i]["TD3CoL"]
        else:
            DDPG_rollouts = np.vstack((DDPG_rollouts, stats_per_situation[i]["DDPG"]))
            CoL_rollouts = np.vstack((CoL_rollouts, stats_per_situation[i]["CoL"]))
            TD3CoL_rollouts = np.vstack((TD3CoL_rollouts, stats_per_situation[i]["TD3CoL"]))

    print(DDPG_rollouts.shape)
    print(CoL_rollouts.shape)
    print(TD3CoL_rollouts.shape)
    df = pd.DataFrame({'Scenarios':situations,\
                      'DDPG': DDPG_rollouts.squeeze(),
                      'CoL': CoL_rollouts.squeeze(),
                      'TD3CoL': TD3CoL_rollouts.squeeze()})
    df = df[['Scenarios','DDPG','CoL', 'TD3CoL']]

    means = df.groupby(["Scenarios"]).mean()
    median = df.groupby(["Scenarios"]).median()
    stds = df.groupby(["Scenarios"]).std()
    print(means)
    print(stds)
    print(median)

    dd=pd.melt(df,id_vars=['Scenarios'],value_vars=['DDPG','CoL', 'TD3CoL'],var_name='Algorithms')

    ax = sns.boxplot(x='Scenarios',y='value',data=dd,hue='Algorithms')
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_xlabel("Scenarios", fontsize=14, fontweight="bold")
    ax.set_ylabel("Speed[Km/Hr]", fontsize=14, fontweight="bold")

    for i, box in enumerate(ax.artists):
        print(i)
        if i in [1, 4, 7, 10, 13, 16, 19]:
            box.set_facecolor('green')
        elif i in [2, 5, 8, 11, 14, 17, 20]:
            box.set_facecolor('blue')

    plt.show()


def success_plots(list_of_data,
                n_situations,
                n_exo_agents,
                title,
                subplot_titles,
                only_goal=False,
                smooth=20,
                add_compound_agg=False):

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
        # ax.set_xlabel("Episodios", fontsize=12, fontweight="bold")
        # ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(subplot_titles[i], fontsize=12, fontweight="bold")
        ax.set_ylim((-0.05, 1.05))
    axes[0].legend(loc="lower left")
    fig.text(0.5, 0.04, 'Episodios', ha='center', fontsize=12, fontweight="bold")
    fig.text(0.08, 0.5, 'Radio de éxito', va='center', rotation='vertical', fontsize=12, fontweight="bold")
    plt.show()

def reward_plots(list_of_data,
                n_situations,
                n_exo_agents,
                title,
                subplot_titles,
                smooth=20,
                add_compound_agg=False):

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
            # ax.set_xlabel("Episodios", fontsize=12, fontweight="bold")
            # ax.set_ylabel(title, fontsize=12, fontweight="bold")
            ax.set_title(subplot_titles[i], fontsize=12, fontweight="bold")
            ax.set_ylim((-1000, 1000))
        axes[0].legend(loc="lower left")
        fig.text(0.5, 0.04, 'Episodios', ha='center', fontsize=12, fontweight="bold")
        fig.text(0.08, 0.5, 'Recompensa promedio', va='center', rotation='vertical', fontsize=12, fontweight="bold")
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
            get_data("2021-11-05-11-44", "TD3CoL", "reward", "test"),
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
            get_data("2021-11-05-11-44", "TD3CoL", "success", "test"),
            get_data("2021-10-21-14-23", "TD3CoL", "success", "test")
            ]
        }, # 3 con 2 en mov
    ]

    escenarios = ["Static", "2Cars1LeadM", "2Cars2RandomM", "3Cars3RandomM", "4Cars4RandomM",
                    "3Cars2StraightM", "3Cars3LeadM"]
    eval_data = [

        {
            "DDPG": ["2021-11-16-17-34"],
            "CoL": ["2021-11-16-13-44"],
            "TD3CoL": ["2021-11-16-16-06"]
        },
        {
            "DDPG": ["2021-11-12-21-30"],
            "CoL": ["2021-11-16-23-47"],
            "TD3CoL": ["2021-11-16-22-44"]
        },
        {
            "DDPG": ["2021-11-17-15-48"],
            "CoL": ["2021-11-17-09-08"],
            "TD3CoL": ["2021-11-14-16-26"]
        },
        {
            "DDPG": ["2021-11-17-13-18"],
            "CoL": ["2021-11-17-20-24"],
            "TD3CoL": ["2021-11-17-17-58"]
        },
        {
            "DDPG": ["2021-11-17-23-49"],
            "CoL": ["2021-11-17-21-36"],
            "TD3CoL": ["2021-11-17-22-37"]
        },
        {
            "DDPG": ["2021-11-18-09-46"],
            "CoL": ["2021-11-18-13-38"],
            "TD3CoL": ["2021-11-18-15-01"]
        },
        {
            "DDPG": ["2021-11-18-19-45"],
            "CoL": ["2021-11-18-18-45"],
            "TD3CoL": ["2021-11-18-16-35"]
        }
    ]

    n_exo_agents = [0, 1, 2, 3, 31, 32]
    titles = ["Sin Exo-vehiculo",
            "1 exo vehiculo",
            "2 exo-vehiculos",
            "3 exo-vehiculos",
            "3 exo-vehiculos con 1 en movimiento",
            "3 exo-vehiculos con 2 en movimiento"]

    # reward_plots(test_reward_data,
    #             n_exo_agents = n_exo_agents,
    #             n_situations=len(n_exo_agents),
    #             title="Recompensa Promedid",
    #             subplot_titles = titles,
    #             add_compound_agg=True)
    # success_plots(test_success_data,
    #                 n_exo_agents = n_exo_agents,
    #                 n_situations=len(n_exo_agents),
    #                 title="Radio de exito",
    #                 subplot_titles = titles,
    #                 add_compound_agg=True)
    # best_choosing(test_success_data, only_goal=False)



    # test_collision_plots(eval_data, escenarios)
    stats_plots(eval_data, escenarios)
    # plot_escenarios()
