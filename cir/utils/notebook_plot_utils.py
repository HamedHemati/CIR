import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb


def plot_single_scenario_alone(scenario_table, n_samples_per_exp, title):
    """
    Plots the heatmap of a given scenario and a line plot of the number of
    sample per experience in the stream.

    :param scenario_table: a C x E tensor (C: # of classes, E: # of experiences)
    :param n_samples_per_exp: a list containing the number of samples in each
        experience.
    :param title: Super title of the plot
    :return:
    """

    # Set General plot settings
    sns.set_theme()

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    plt.suptitle(title, fontsize=20, y=1.02, wrap=True)
    fig.tight_layout()

    # Colors
    n_classes = scenario_table.shape[0]
    cmap = sns.color_palette("deep", n_classes+1)
    cmap[0] = (1, 1, 1)

    # Heat Map
    table_plot = scenario_table.clone()
    for i in range(table_plot.shape[0]):
        idx = table_plot[i] > 0.0
        table_plot[i][idx] = i + 1
    sns.heatmap(table_plot, cbar=False, cmap=cmap, vmin=0, ax=ax)


def plot_single_scenario(scenario_table, n_samples_per_exp, title):
    """
    Plots the heatmap of a given scenario and a line plot of the number of
    sample per experience in the stream.

    :param scenario_table: a C x E tensor (C: # of classes, E: # of experiences)
    :param n_samples_per_exp: a list containing the number of samples in each
        experience.
    :param title: Super title of the plot
    :return:
    """

    # Set General plot settings
    sns.set_theme()

    fig, ax = plt.subplots(2, 1, figsize=(20, 10),
                           gridspec_kw={'height_ratios': [4, 2]})
    plt.suptitle(title, fontsize=20, y=1.02, wrap=True)
    fig.tight_layout()

    # Colors
    n_classes = scenario_table.shape[0]
    cmap = sns.color_palette("deep", n_classes+1)
    cmap[0] = (1, 1, 1)

    # Heat Map
    table_plot = scenario_table.clone()
    for i in range(table_plot.shape[0]):
        idx = table_plot[i] > 0.0
        table_plot[i][idx] = i + 1
    sns.heatmap(table_plot, cbar=False, cmap=cmap, vmin=0, ax=ax[0])

    # LinePlot
    n_exp = scenario_table.shape[1]
    total_samples = [n_samples_per_exp[i] for i in range(n_exp)]
    sample_data = {"N_Samples": total_samples,
                   "Type": ["Total"] * n_exp ,
                   "Experience": list(range(n_exp))}

    sns.lineplot(data=sample_data, x="Experience", y="N_Samples",
                            hue="Type", ax=ax[1], legend=True, marker="o",
                            markers=["o", "o", "o", "o"])

    ax[1].set(ylim=(0, None))


def print_scenario_details(scenario):
    coverage_perc = float(scenario.n_unique_samples) / scenario.n_trainset
    repetition_perc = float(scenario.n_total_samples -
                            scenario.n_unique_samples) / scenario.n_trainset
    print("Coverage: ", coverage_perc)
    print("Percentage of old instance repetition: ", repetition_perc)


def plot_scenario_transition(
        scenario_tables,
        n_samples_per_exp_list,
        title,
        values
):

    fig, ax = plt.subplots(nrows=2, ncols=len(scenario_tables), figsize=(15, 3))
    plt.suptitle(title, fontsize=15)
    fig.tight_layout()

    for k in range(len(scenario_tables)):
        scenario_table = scenario_tables[k]
        n_samples_per_exp = n_samples_per_exp_list[k]
        # Data
        n_classes = scenario_table.shape[0]
        cmap = sns.color_palette("deep", n_classes+1)
        cmap[0] = (1, 1, 1)

        # Heat Map
        table_plot = scenario_table.clone()
        for i in range(table_plot.shape[0]):
            idx = table_plot[i] > 0.0
            table_plot[i][idx] = i + 1
        sns.heatmap(table_plot, cbar=False, cmap=cmap, vmin=0, ax=ax[0, k])

        # LinePlot
        n_exp = scenario_table.shape[1]
        total_samples = [n_samples_per_exp[i] for i in range(n_exp)]
        sample_data = {"N_Samples": total_samples,
                       "Type": ["Total"] * n_exp,
                       "Experience": list(range(n_exp))}

        sns.lineplot(data=sample_data, x="Experience", y="N_Samples",
                     hue="Type", ax=ax[1, k], legend=False, marker="o",
                     markers=["o", "o", "o", "o"])

        ax[1, k].set(ylim=(0, None))

        ax[0, k].set_title(f"Value: {values[k]}")

    plt.gcf().subplots_adjust(bottom=0.15)


##########################################
#             WANDB utils
##########################################


def get_wandb_project_runs(entity="", project=""):
    """ Retrieves configs for all runs in a project. """
    api = wandb.Api()
    runs = api.runs(entity + "/" + project)

    def get_run_config(run):
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        return config

    run_configs = {run.name: get_run_config(run) for run in runs if
                   get_run_config(run) is not None}

    return run_configs


def read_exported_panel_csv(file_path):
    """ Reads exported data from a csv file. """
    data = pd.read_csv(file_path)
    data = data.to_dict()
    data = {k: v for k, v in data.items() if
            not k.endswith(("__MIN", "__MAX", "_step"))}
    data = {k.split(" - ")[0].strip(): v for (k, v) in data.items()}

    # Filter data if needed
    panel_data = {}
    for k in data.keys():
        panel_data[k] = data[k]

    print("List of keys: ", list(panel_data.keys()))

    return panel_data
