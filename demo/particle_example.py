from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import yaml
from dust.controllers.disco import MultiDISCO
from dust.inference import likelihoods
from dust.inference.mpf import MPF
from dust.inference.svgd import get_gmm
from dust.inference.svmpc import SVMPC
from dust.kernels.base_kernels import RBF
from dust.kernels.composite_kernels import iid_mp
from dust.models.particle import Particle
from dust.utils.helper import create_video_from_plots, save_progress
from gpytorch.kernels import RBFKernel
from tqdm import trange


def main(sim_params, exp_params, env_params):
    # ========== SIMULATION HYPERPARAMETERS ==========
    WARM_UP = sim_params["warm_up"]
    STEPS = sim_params["steps"]
    EPISODES = sim_params["episodes"]
    # ========== EXPERIMENT HYPERPARAMETERS ==========
    HORIZON = exp_params["horizon"]
    N_PARTICLES = exp_params["n_particles"]
    ACTION_SAMPLES = exp_params["action_samples"]
    PARAMS_SAMPLES = exp_params["params_samples"]
    ALPHA = exp_params["alpha"]
    LEARNING_RATE = exp_params["learning_rate"]
    BANDWIDTH_SCALE = exp_params["bandwidth_scaling"]
    CTRL_SIGMA = exp_params["ctrl_sigma"]
    CTRL_DIM = exp_params["ctrl_dim"]
    LIKELIHOOD = exp_params["likelihood"]
    USE_SVMPC = exp_params["use_svmpc"]
    USE_MPF = exp_params["use_mpf"]
    PRIOR_SIGMA = exp_params["prior_sigma"]
    WEIGHTED_PRIOR = exp_params["weighted_prior"]
    DYN_PRIOR = exp_params["dyn_prior"]
    DYN_PRIOR_ARG1 = exp_params["dyn_prior_arg1"]
    DYN_PRIOR_ARG2 = exp_params["dyn_prior_arg2"]
    LOAD = exp_params["extra_load"]
    SAMPLING = exp_params["sampling"]
    # ========== Experiment Setup ==========
    # Initial state
    state = torch.as_tensor(env_params["init_state"]).clone()
    policies_prior = get_gmm(
        torch.randn(N_PARTICLES, HORIZON, CTRL_DIM),
        torch.ones(N_PARTICLES),
        PRIOR_SIGMA ** 2 * torch.eye(CTRL_DIM),
    )
    init_policies = policies_prior.sample([N_PARTICLES])
    dynamics_prior = getattr(dist, DYN_PRIOR)(DYN_PRIOR_ARG1, DYN_PRIOR_ARG2)

    system_kwargs = {
        "uncertain_params": ["mass"],
        "mass": dynamics_prior.mean,
    }
    # Model is used for the internal rollouts, system is the simulator,
    # they may have different parameters.
    base_system = Particle(**env_params, **system_kwargs)
    base_model = Particle(**env_params, **system_kwargs)

    base_controller = MultiDISCO(
        base_model.observation_space,
        base_model.action_space,
        HORIZON,
        N_PARTICLES,
        ACTION_SAMPLES,
        temperature=1 / ALPHA,
        a_cov=CTRL_SIGMA ** 2 * torch.eye(CTRL_DIM),
        params_sampling=SAMPLING,
        params_samples=PARAMS_SAMPLES,
        params_log_space=exp_params["mpf_log_space"],
        inst_cost_fn=base_model.default_inst_cost,
        term_cost_fn=base_model.default_term_cost,
    )

    # Define a factorized additive Kernel were we compute distances for each
    # timestep independently
    # def factorized_kernel(x, h):
    #     base_kernel = RBFKernel()
    #     base_kernel.lengthscale = h
    #     K = AdditiveStructureKernel(base_kernel, x.size(0))(x)
    #     return K.evaluate() / x.size(-1)
    #
    # def rbf_kernel(x, h):
    #     base_kernel = RBFKernel()
    #     base_kernel.lengthscale = h
    #     return base_kernel(x).evaluate()
    #
    # kernel = factorized_kernel if FACTORIZE_KERNEL else rbf_kernel

    kernel_type = exp_params["kernel"]
    if kernel_type == "message_passing":
        base_kernel = RBF(bandwidth=-1)
        kernel = iid_mp(base_kernel=base_kernel, ctrl_dim=2, indep_controls=True)
    elif kernel_type == "rbf":
        kernel = RBFKernel()
    else:
        raise ValueError("Kernel type '{}' is not valid.".format(kernel_type))
    policies_lik = getattr(likelihoods, LIKELIHOOD)(
        ALPHA, controller=base_controller, model=base_model, n_samples=ACTION_SAMPLES
    )
    svmpc_kwargs = {
        "init_particles": init_policies.detach().clone(),
        "prior": policies_prior,
        "likelihood": policies_lik,
        "kernel": kernel,
        "n_particles": N_PARTICLES,
        "bw_scale": BANDWIDTH_SCALE,
        "n_steps": 1,
        "optimizer_class": torch.optim.SGD,
        "lr": LEARNING_RATE,
        "weighted_prior": WEIGHTED_PRIOR,
    }
    # TODO: Implement new SVMPC with indep. gradients along horizon
    base_svmpc = SVMPC(**svmpc_kwargs)

    mpf_n_part = exp_params["mpf_n_particles"]
    mpf_steps = exp_params["mpf_steps"]
    mpf_log_space = exp_params["mpf_log_space"]
    mpf_bw = exp_params["mpf_bandwidth"]
    mpf_init = dynamics_prior.sample([mpf_n_part, 1]).clamp(min=1e-6)
    mpf_init = mpf_init.log() if mpf_log_space else mpf_init
    dynamics_lik = likelihoods.GaussianLikelihood(
        initial_obs=state,
        obs_std=exp_params["mpf_obs_std"],
        model=base_model,
        log_space=exp_params["mpf_log_space"],
    )
    base_mpf = MPF(
        init_particles=mpf_init,
        likelihood=dynamics_lik,
        optimizer_class=torch.optim.SGD,
        lr=exp_params["mpf_learning_rate"],
        bw=(2 * exp_params["dyn_prior_arg2"]) ** 1 / 2,
        bw_scale=exp_params["mpf_bandwidth_scaling"],
    )

    base_path = save_progress(params=config_data)
    # x_lim = system_kwargs["mass"] + 2 + LOAD
    x_lim = 5
    lin_s = torch.linspace(0, x_lim, 100)
    log_s = lin_s.log()

    # ===== Experiment Loop =====
    for ep in range(EPISODES):
        # Reset state
        state = torch.as_tensor(env_params["init_state"]).clone()
        tau = state.unsqueeze(0)
        rollouts = torch.empty(
            0,
            base_controller.n_params,
            base_controller.n_actions,
            base_controller.n_pol,
            base_controller.hz_len + 1,
            state.shape[0],
        )
        costs = torch.empty((0, 1))
        actions = torch.empty((0, CTRL_DIM))
        dyn_particles = torch.empty((0, mpf_n_part))
        iterator = trange(STEPS)
        system = deepcopy(base_system)
        model = deepcopy(base_model)
        svmpc = deepcopy(base_svmpc)
        if USE_MPF:
            mpf = deepcopy(base_mpf)
            dynamics_dist = mpf.prior
        else:
            mpf = None
            dynamics_dist = dynamics_prior
        controller = deepcopy(base_controller)
        save_path = save_progress(folder_name=base_path.stem + "/ep{}".format(ep))
        for step in iterator:
            if step == STEPS // 4:  # Changes the simulator mass
                system.params_dict["mass"] = system.params_dict["mass"].clone() + LOAD
            if USE_SVMPC is True:
                svmpc.optimize(state, dynamics_dist)
                if step < WARM_UP:
                    action = torch.zeros(CTRL_DIM)
                else:
                    # we now re-sample the likelihood to get the expected cost of
                    # each new θ_i. note, this will call the forward function of
                    # the controller.
                    a_seq, _ = svmpc.forward(state, dynamics_dist)
                    action = a_seq[0]
                states = svmpc.likelihood.last_states.detach().clone()
            else:
                _, states, _, _, _ = controller.forward(
                    state, model, params_dist=dynamics_dist
                )
                action = controller.step(strategy="argmax").flatten()

            # Rollout dynamics, get costs
            # ----------------------------
            # selects next action and makes sure the controller plan is
            # the same as svmpc
            state = system.step(state, action.squeeze())
            grad_dyn = torch.zeros(mpf_steps)
            if step >= WARM_UP and mpf is not None:
                # optimize will automatically update `dynamics_dist`
                grad_dyn, _ = mpf.optimize(
                    action.squeeze(), state, bw=mpf_bw, n_steps=mpf_steps
                )

            tau = torch.cat([tau, state.unsqueeze(0)], dim=0)
            actions = torch.cat([actions, action.unsqueeze(0)])
            rollouts = torch.cat([rollouts, states.unsqueeze(0)], dim=0)
            inst_cost = controller.inst_cost_fn(state.view(1, -1))
            costs = torch.cat([costs, inst_cost.unsqueeze(0)], dim=0)
            if USE_MPF:
                dyn_particles = torch.cat(
                    [dyn_particles, mpf.x.detach().clone().t()], dim=0
                )

            ax3 = plt.subplot(224)
            ax3.plot(grad_dyn)
            ax2 = plt.subplot(222)
            ax2.set_xlim(0, x_lim)
            if USE_MPF:
                ax2.plot(lin_s, mpf.prior.log_prob(log_s.unsqueeze(1)).exp().numpy())
            ax2.axvline(system.params_dict["mass"], ls="--", c="r")
            ax1 = plt.subplot(121)
            ax1.set_title("Sampling: {}".format(SAMPLING))
            system.render(
                path=save_path / "plots/{0:03d}.png".format(step),
                states=tau[:, :2],
                rollouts=states.flatten(0, 1),  # flattens actions and params samples
                ax=ax1,
            )
            plt.close()
            if system.with_obstacle:
                if system.obst_map.get_collisions(state[:2]):
                    print("\nCrashed at step {}".format(step))
                    # print("Last particles state:")
                    # print(svmpc.theta)
                    break
            if (system.target - state).norm() <= 1.0:
                break

        episode_data = {
            "costs": costs,
            "trajectory": tau,
            "actions": actions,
            "rollouts": rollouts,
            "dyn_particles": dyn_particles,
        }
        save_progress(
            folder_name=save_path.relative_to(base_path.parent), data=episode_data
        )
        create_video_from_plots(save_path)


if __name__ == "__main__":
    config_file = Path("demo/particle_config.yaml")
    with config_file.open() as f:
        config_data = yaml.load(f, yaml.FullLoader)

    sim_params = {}
    try:
        sim_params = config_data["sim_params"]
    except KeyError:
        print("Invalid key for simulation params!")

    exp_params = {}
    try:
        exp_params = config_data["exp_params"]
    except KeyError:
        print("Invalid key for experiment params!")

    env_params = {}
    try:
        env_params = config_data["env_params"]
    except KeyError:
        print("Invalid key for environment params!")

    main(sim_params, exp_params, env_params)
