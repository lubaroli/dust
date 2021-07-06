from copy import deepcopy

import gym
import pandas as pd
import torch
from tqdm import trange

from ..inference.likelihoods import ExponentiatedUtility
from ..inference.svmpc import SVMPC
from ..models.pendulum import PendulumModel


def run_pendulum_simulation(
    init_state,
    init_policies,
    model_kwargs,
    dyn_dist,
    experiment_params,
    controller,
    use_exact_model=True,
    use_svmpc=True,
    svmpc_kwargs=None,
    lik_kwargs=None,
    mpf=None,
    mpf_bw=None,
    mpf_steps=20,
    episodes=3,
    steps=200,
    render=False,
    warm_up=1,
    verbose=False,
    steps_per_message=20,
):
    epoch_df = pd.DataFrame()

    for i in range(episodes):
        print("--- Starting iteration {} ---".format(i + 1))
        # Dimensions are: Samples, Particles, Horizon, State/Action space
        # States are position and angular velocity (x and x_dot)

        # reset iteration variables
        step = 0
        if use_exact_model:
            model = PendulumModel(**experiment_params[i], **model_kwargs)
        else:
            model = PendulumModel(
                length=dyn_dist.mean[0], mass=dyn_dist.mean[1], **model_kwargs
            )
        env = gym.make("Pendulum-v0")
        env.reset()
        env.unwrapped.l = experiment_params[i]["length"]
        env.unwrapped.m = experiment_params[i]["mass"]
        env.unwrapped.state = init_state
        state = init_state.unsqueeze(0)
        print(
            "parameters are: {0} {2:.2f} and {1} {3:.2f}".format(
                *experiment_params[i].keys(), *experiment_params[i].values()
            )
        )
        # make sure svmpc and multidisco have the same starting actions, but
        # don't point to the same variable!
        sim_ctrl = deepcopy(controller)
        sim_ctrl.a_mat = init_policies.detach().clone()
        sim_svmpc = None
        if use_svmpc:
            assert (
                svmpc_kwargs is not None and lik_kwargs is not None
            ), "Need a Stein Optimizer and likelihood for dual svmpc simulation."
            likelihood = ExponentiatedUtility(
                **lik_kwargs, controller=sim_ctrl, model=model,
            )
            sim_svmpc = SVMPC(likelihood=likelihood, **svmpc_kwargs)

        sim_mpf = None
        dyn_particles = None
        dyn_bws = None
        if mpf is not None:
            sim_mpf = deepcopy(mpf)
            dyn_dist = sim_mpf.prior
            dyn_particles = torch.full(
                (steps, *sim_mpf.x.size()), fill_value=float("nan"), dtype=torch.float,
            )
            dyn_bws = torch.zeros(steps, dtype=torch.float)

        # create the aggregating tensors and set them to 'nan' to check if
        # simulation breaks
        states = torch.full(
            (steps, sim_ctrl.dim_s), fill_value=float("nan"), dtype=torch.float,
        )
        actions = torch.full(
            (steps, sim_ctrl.dim_a), fill_value=float("nan"), dtype=torch.float,
        )
        costs = torch.full((steps, 1), fill_value=float("nan"), dtype=torch.float)
        pol_particles = torch.full(
            (steps, sim_ctrl.n_pol, sim_ctrl.hz_len, sim_ctrl.dim_a),
            fill_value=float("nan"),
            dtype=torch.float,
        )
        weights = torch.full(
            (steps, sim_ctrl.n_pol), fill_value=float("nan"), dtype=torch.float,
        )

        # main iteration loop
        for step in range(steps):
            if render:
                env.render()

            if use_svmpc:
                sim_svmpc.optimize(state, dyn_dist)
                if step < warm_up:
                    action = torch.zeros(sim_ctrl.dim_a)
                else:

                    # we now re-sample the likelihood to get the expected cost of
                    # each new θ_i. note, this will call the forward function of
                    # the controller.
                    a_seq, p_weights = sim_svmpc.forward(state, dyn_dist)
                    action = a_seq[0]
                    # selects next action and makes sure the controller plan is
                    # the same as svmpc
                    # action = sim_ctrl.step(strategy="external", ext_actions=a_seq)
                    pol_particles[step] = sim_svmpc.theta.detach().clone()
                    weights[step] = p_weights
            else:  # if not using svmpc, use disco
                sim_ctrl.forward(state, model, dyn_dist)
                action = sim_ctrl.step(strategy="average").flatten()

            actions[step] = action
            _, _, done, _ = env.step(action)
            state = torch.as_tensor(env.state, dtype=torch.float).unsqueeze(0)

            if sim_mpf is not None:
                # optimize will automatically update `dynamics_dist`
                _, bw = sim_mpf.optimize(
                    action.squeeze(), state, bw=mpf_bw, n_steps=mpf_steps
                )
                dyn_particles[step] = sim_mpf.x
                dyn_bws[step] = bw
                # print(
                #     f"Mean: {list(sim_mpf.x.mean(0))}   Std: {list(sim_mpf.x.std(0))}"
                # )

            cost = sim_ctrl.inst_cost_fn(state.view(1, -1))

            if verbose and not step % steps_per_message:
                print(
                    "Step {0}: action taken {1:.2f}, cost {2:.2f}".format(
                        step, float(action), float(cost)
                    )
                )
                print(
                    "Current state: theta={0[0]}, theta_dot={0[1]}".format(
                        state.squeeze()
                    )
                )
            states[step] = state
            costs[step] = cost
            if done:
                env.close()
                break

        # End of episode
        env.close()
        if verbose:
            print(
                "Last step {0}: action taken {1:.2f}, cost {2:.2f}".format(
                    step, float(action), float(cost)
                )
            )
            print("Last state: theta={0[0]}, theta_dot={0[1]}".format(state.squeeze()))

        episode_df = pd.DataFrame(
            index=list(range(200)),
            data={
                "Cost": costs[:, 0],
                "Position": states[:, 0],
                "Speed": states[:, 1],
                "Actions": actions[:, 0],
                "Timestep": torch.arange(200),
                "Iteration": i,
                "DynParticles": dyn_particles.tolist()
                if dyn_particles is not None
                else dyn_particles,
                "DynBandwidths": dyn_bws,
                "PolParticles": pol_particles[..., 0, 0].tolist(),
                "Weights": weights.tolist(),
                "ExpParams": 200 * [list(experiment_params[i].values())],
            },
        )
        episode_df["AvgCumCost"] = (
            episode_df["Cost"].cumsum(0) / (episode_df["Timestep"] + 1)
        ).round(2)
        epoch_df = pd.concat((epoch_df, episode_df), axis=0)
    return epoch_df


def run_particle_episode(
    init_state,
    model,
    dyn_dist,
    controller,
    use_svmpc=True,
    warm_up=30,
    svmpc: SVMPC = None,
    load=0,
    steps=400,
    render=False,
    save_path=None,
):
    """Uses initial controller model as simulation system before altering the load.
    """
    system = deepcopy(model)
    tau = init_state[:2].unsqueeze(0)
    cum_cost = 0
    state = init_state
    iterator = trange(steps)
    for step in iterator:
        if step == steps // 4:  # Changes the simulator mass
            system.params_dict["mass"] += load
        if use_svmpc is True:
            svmpc.optimize(state, dyn_dist)
            if step < warm_up:
                action = torch.zeros(controller.dim_a)
            else:
                # we now re-sample the likelihood to get the expected cost of
                # each new θ_i. note, this will call the forward function of
                # the controller.
                a_seq, _ = svmpc.forward(state, dyn_dist)
                action = a_seq[0]
            states = svmpc.likelihood.last_states.detach().clone()
        else:
            _, states, _, _, _ = controller.forward(state, model, params_dist=dyn_dist)
            action = controller.step(strategy="argmax")

        # Rollout dynamics, get costs
        # ----------------------------
        # selects next action and makes sure the controller plan is
        # the same as svmpc
        state = system.step(state, action.squeeze())
        cost = controller.inst_cost_fn(state.view(1, -1))
        cum_cost += cost

        tau = torch.cat([tau, state[:2].unsqueeze(0)], dim=0)
        if render:
            system.render(
                path=save_path / "plots/{0:03d}.png".format(step),
                state=tau,
                rollouts=states,
            )
        if system.with_obstacle:
            if system.obst_map.get_collisions(state[:2]):
                print("Crashed at step {}".format(step))
                print("Last particles state:")
                print(svmpc.theta)
                cum_cost = float("inf")
                break
        if (system.target - state).norm() <= 1.0:
            break

    return cum_cost
