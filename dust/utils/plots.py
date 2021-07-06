import sys
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.distributions as dist
import yaml


def _label(data, color, label):
    """Define and use a simple function to label the plot in axes coordinates.
    """
    ax = plt.gca()
    ax.text(
        0,
        0.2,
        "Step " + label,
        fontweight="bold",
        fontsize=20,
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


def _create_df(data, gmm_std_dev, steps, n_plots):
    if steps is None:
        steps = torch.arange(0, data.shape[0], int(data.shape[0] / n_plots))
        steps = torch.cat([steps, torch.as_tensor(data.shape[0] - 1).unsqueeze(0)])
    else:
        steps = torch.as_tensor(steps)
        steps = steps[steps < len(data)]
    particles = data[steps]
    x_min = torch.floor(particles.min() - 0.25)
    x_max = torch.ceil(particles.max() + 0.25)
    mix = dist.Categorical(torch.ones_like(particles))
    comp = dist.Normal(
        loc=particles, scale=gmm_std_dev ** 2 * torch.ones_like(particles)
    )
    densities = dist.MixtureSameFamily(mix, comp)
    x = torch.arange(x_min, x_max, 0.01)
    probs = densities.log_prob(x.repeat(len(steps), 1).T).exp().T
    # pts = particles.shape[1]
    pts = x.shape[0]
    long_steps = steps.repeat(pts).reshape(pts, -1).T.flatten()
    # creates a long-format DataFrame (un-pivoted)
    df = pd.DataFrame(
        dict(data=probs.flatten(), x=x.repeat(steps.shape), step=long_steps)
    )
    return df, steps


def _load_config_data(path):
    assert (
        path.is_dir() is True
    ), "Path needs to point to root dir of batch simulations."

    try:
        config_file = Path(path / "config.yaml")
        with config_file.open() as f:
            config_data = yaml.load(f, yaml.FullLoader)
    except IOError as e:
        sys.exit("Couldn't load experiment configuration file from {}.".format(e))
    return config_data


def plot_mean_results(
    data_path: Path,
    title: str,
    y_key: str,
    y_label: str,
    x_key: str = "Timestep",
    x_label: str = "Timestep",
    color_key: str = "Case",
):
    try:
        source = torch.load(data_path / "data.pkl")
        source = pd.DataFrame(source)
        source["CumCost"] = source["AvgCumCost"] * (source["Timestep"] + 1)
        columns = ["Case", y_key, x_key]
        source = source[columns]
    except IOError as e:
        print("Couldn't load data file.")
        sys.exit(e)

    # # Remove comment block if you want to add an error band to the plot
    # band = (
    #     alt.Chart(source)
    #     .mark_errorband(extent="stdev")
    #     .encode(
    #         alt.Y("mean({}):Q".format(y_key), title=y_label),
    #         alt.X("{}:Q".format(x_key), title=x_label),
    #         color="{}:N".format(color_key),
    #         opacity=alt.value(0.3),
    #     )
    # )

    line = (
        alt.Chart(source)
        .mark_line(interpolate="basis", clip=True)
        .encode(
            alt.Y(
                "mean({}):Q".format(y_key),
                title=y_label,
                # scale=alt.Scale(domain=[0, 10000]),
            ),
            alt.X("{}:Q".format(x_key), title=x_label),
            color=alt.Color(
                "{}:N".format(color_key), scale=alt.Scale(scheme="category10")
            ),
        )
    )

    # Create a selection that chooses the nearest point & selects based on
    # x-value
    nearest = alt.selection(
        type="single",
        nearest=True,
        on="mouseover",
        fields=["{}".format(x_key)],
        empty="none",
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = (
        alt.Chart(source)
        .mark_point()
        .encode(x="{}:Q".format(x_key), opacity=alt.value(0),)
        .add_selection(nearest)
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align="left", dx=5, dy=-5).encode(
        text=alt.condition(nearest, "mean({}):Q".format(y_key), alt.value(" "))
    )

    # Draw a rule at the location of the selection
    rules = (
        alt.Chart(source)
        .mark_rule(color="gray")
        .encode(x="{}:Q".format(x_key),)
        .transform_filter(nearest)
    )
    chart = (
        alt.layer(line, selectors, points, rules, text)
        .properties(title=title, width=1000, height=400)
        .interactive()
    )

    plot_path = data_path / "plots/"
    if not plot_path.exists():
        plot_path.mkdir()
    chart.save(data_path.as_posix() + "/plots/{}_plot.html".format(y_key.lower()))


def plot_stein_particles(source, n_particles):
    col_names = ["p{}".format(idx) for idx in range(n_particles)]
    particles_df = pd.DataFrame(source.Particles.tolist(), columns=col_names,)
    particles_df["Timestep"] = source.Timestep.reset_index(drop=True)
    particles_df = particles_df.melt(
        id_vars=["Timestep"], var_name="Particle", value_name="Value"
    )
    weight_df = pd.DataFrame(source.Weights.tolist())
    weight_df["Timestep"] = source.Timestep.reset_index(drop=True)
    weight_df = weight_df.melt(id_vars=["Timestep"], value_name="Weight")
    particles_df = particles_df.join(weight_df.Weight)
    part_chart = (
        alt.Chart(particles_df)
        .mark_circle(opacity=0.5)
        .encode(
            alt.X("Timestep"),
            alt.Y("Value", scale=alt.Scale(domain=(-3, 3))),
            color="Particle:N",
            size="Weight",
        )
    )
    nearest = alt.selection(
        type="single", nearest=True, on="mouseover", fields=["Timestep"], empty="none",
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = (
        alt.Chart(particles_df)
        .mark_point()
        .encode(x="Timestep", opacity=alt.value(0),)
        .add_selection(nearest)
    )

    # Draw a rule at the location of the selection
    rules = (
        alt.Chart(particles_df)
        .mark_rule(color="gray")
        .encode(x="Timestep",)
        .transform_filter(nearest)
    )
    composed_chart = (
        alt.layer(part_chart, selectors, rules)
        .properties(width=1000, height=400)
        .interactive()
    )
    return composed_chart


def plot_part2d_cost(paths: list, labels: list, kwargs={}, cmap="Blues"):
    # assume all simulations have the same length
    config_data = _load_config_data(paths[0])
    try:
        sim_length = config_data["sim_params"]["steps"]
    except KeyError:
        msg = "Couldn't load number of simulation steps!"
        sys.exit(msg)

    sns.set_theme(context="paper", palette="colorblind")
    sns.despine()
    for path, label in zip(paths, labels):
        cost_batch = torch.tensor([])
        for ep_path in [p for p in path.iterdir() if p.is_dir() and p.stem != "plots"]:
            try:
                data = torch.load(ep_path / "data.pkl")
                costs = torch.as_tensor(data["costs"])
                ep_length = costs.shape[0]
                if ep_length < sim_length:
                    # repeat penultimate cost until the end of the episode
                    res = torch.ones(sim_length) * costs[-2]
                    res[: ep_length - 1] = costs[:-1].flatten()
                else:
                    res = costs
                cost_batch = torch.cat([cost_batch, res.view(1, -1)], 0)
            except IOError:
                print("Couldn't load data file from '{}'.".format(ep_path / "data.pkl"))
        if cost_batch.shape[0] > 0:
            mean_cost = cost_batch.mean(0)
            std_cost = cost_batch.std(0)
            plt.plot(mean_cost, label=label, **kwargs)
            plt.fill_between(
                torch.arange(0, 400, 1),
                mean_cost - std_cost,
                mean_cost + std_cost,
                alpha=0.3,
            )

    plt.legend()
    plt.gca().set_xlim(left=0, right=400)
    plt.xlabel("Step")
    plt.ylabel("Cost")
    plt.tight_layout()
    save_path = paths[0].parent / "part2d_costs.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_part2d_traj(path: Path, traj_kwargs={}, marker_kwargs={}, cmap="binary"):
    from ..models.particle import Particle

    config_data = _load_config_data(path)
    env_params = {}
    try:
        env_params = config_data["env_params"]
    except KeyError:
        print("Invalid key for environment params!")

    system = Particle(**env_params)

    traj_batch = []
    traj_crashed = []
    for ep_path in [p for p in path.iterdir() if p.is_dir() and p.stem != "plots"]:
        try:
            data = torch.load(ep_path / "data.pkl")
            if data["costs"][-1] >= 1e6:  # if crashed
                traj_crashed.append(True)
            else:
                traj_crashed.append(False)
            traj_batch.append(data["trajectory"])
        except IOError:
            print("Couldn't load data file from '{}'.".format(ep_path / "data.pkl"))

    batch = sorted(zip(traj_batch, traj_crashed), key=lambda k: k[1])
    if len(batch) > 0:
        plot_targets = True
        colors = iter(sns.color_palette("colorblind", len(batch)))
        # colors = iter(plt.cm.tab20(range(len(batch))))
        for traj, _ in batch:
            c = next(colors)
            system.render(
                traj, cmap=cmap, color=c, plot_goal=plot_targets, **traj_kwargs
            )
            plot_targets = False  #
        colors = iter(sns.color_palette("colorblind", len(batch)))
        # colors = iter(plt.cm.tab20(range(len(batch))))
        for traj, crashed in batch:
            c = next(colors)
            if crashed:
                system.render(traj[-1], cmap=cmap, color=c, **marker_kwargs)

    plt.xticks([], [])
    plt.yticks([], [])
    sns.despine()
    save_path = path / "plots/trajectories_plot.pdf"
    if not save_path.parent.exists():
        save_path.parent.mkdir()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_dist_ridgeplot(path: Path, gmm_std_dev=0.1, steps=None, n_plots=5):
    """Creates a ridge plot of MPF GMM over specified time steps.
    """
    config_data = _load_config_data(path)
    try:
        log_space = config_data["exp_params"]["mpf_log_space"]
        initial_mass = torch.tensor(
            config_data["exp_params"]["dyn_prior_arg1"], dtype=torch.float32
        )
        final_mass = initial_mass + config_data["exp_params"]["extra_load"]
        switchover = config_data["exp_params"]["steps"] / 4
    except KeyError as e:
        sys.exit("Invalid configuration file key: {}.".format(e))

    data_batch = []
    for ep_path in [p for p in path.iterdir() if p.is_dir() and p.stem != "plots"]:
        try:
            ep_data = torch.load(ep_path / "data.pkl")
        except IOError as e:
            print("Couldn't load data file.")
            sys.exit(e)
        if log_space:
            data_batch.append(ep_data["dyn_particles"].exp())
        else:
            data_batch.append(ep_data["dyn_particles"])
        # break  # use only first episode

    # generate plots
    for ep_idx, ep in enumerate(data_batch):
        df, steps = _create_df(ep, gmm_std_dev, steps, n_plots)
        latent_vals = torch.where(steps < switchover, initial_mass, final_mass)

        # Initialize the FacetGrid object
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        pal = sns.cubehelix_palette(len(steps), rot=-0.25, light=0.7)
        grid = sns.FacetGrid(
            df,
            row="step",
            hue="step",
            aspect=10,
            height=1.0,
            palette=pal,
            xlim=(df["x"].min(), df["x"].max()),
        )

        # Draw the densities in a few steps
        grid.map(
            plt.plot, "x", "data", linewidth=1.5,
        )
        grid.map(plt.fill_between, "x", "data", alpha=0.3)
        # grid.map(sns.kdeplot, "data", clip_on=False, color="w", lw=2, bw_adjust=0.5)
        grid.map(plt.axhline, y=0, lw=2, clip_on=False)
        grid.map(_label, "data")
        for idx, ax in enumerate(grid.axes):
            ax[0].axvline(latent_vals[idx], ymax=0.7, lw=1.5, ls="--", c="r", alpha=0.7)

        # Set the subplots to overlap
        grid.fig.subplots_adjust(hspace=-0.25)

        # Remove axes details that don't play well with overlap
        grid.set_titles("")
        grid.set(yticks=[])
        grid.set_xticklabels(fontweight="bold", fontsize=20, color=grid._colors[-1])
        grid.set_xlabels(
            r"Mass ($\mathrm{kg}$)",
            fontweight="bold",
            fontsize=20,
            color=grid._colors[-1],
        )
        grid.despine(bottom=True, left=True)
        save_path = path / "plots/ridge_plot_ep{}.pdf".format(ep_idx)
        if not save_path.parent.exists():
            save_path.parent.mkdir()
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
