import math
from pathlib import Path

import pandas as pd
import torch
import torch.distributions as dist
import yaml
from dust.controllers.disco import MultiDISCO
from dust.inference.likelihoods import GaussianLikelihood
from dust.inference.mpf import MPF
from dust.inference.svgd import get_gmm
from dust.kernels.base_kernels import RBF
from dust.kernels.composite_kernels import iid_mp
from dust.models.pendulum import PendulumModel
from dust.utils.helper import save_progress
from dust.utils.simulations import run_pendulum_simulation
from dust.utils.utf import MerweScaledUTF
from gpytorch.kernels import RBFKernel


def inst_cost(states, controls=None, n_pol=1, debug=None):
    # Note that theta may range beyond 2*pi
    theta, theta_d = states.chunk(2, dim=1)
    return 50.0 * (theta.cos() - 1) ** 2 + 1.0 * theta_d ** 2


def term_cost(states, n_pol=1, debug=None):
    return inst_cost(states).squeeze()


if __name__ == "__main__":
    config_file = Path("demo/pendulum_config.yaml")
    with config_file.open() as f:
        config_data = yaml.load(f, yaml.FullLoader)

    exp_params = config_data["exp_params"]
    sim_kwargs = config_data["sim_params"]

    # ================================================
    # ========== EXPERIMENT HYPERPARAMETERS ==========
    # ================================================
    PI = math.pi
    ONE_DEG = 2 * PI / 360
    HORIZON = exp_params["horizon"]
    N_PARTICLES = exp_params["n_particles"]
    ACTION_SAMPLES = exp_params["action_samples"]
    PARAMS_SAMPLES = exp_params["params_samples"]
    ALPHA = exp_params["alpha"]
    LEARNING_RATE = exp_params["learning_rate"]
    BANDWIDTH_SCALE = exp_params["bandwidth_scaling"]
    CTRL_SIGMA = exp_params["ctrl_sigma"]
    CTRL_DIM = exp_params["ctrl_dim"]
    PRIOR_SIGMA = exp_params["prior_sigma"]
    PARAMS_LOC = exp_params["params_prior_loc"]
    PARAMS_SIGMA = exp_params["params_prior_sigma"]

    # ======================================
    # ========== SIMULATION SETUP ==========
    # ======================================

    # Define environment...
    env_model = PendulumModel()

    init_state = torch.as_tensor(exp_params["init_state"]).clone()
    # Policies prior...
    policies_prior = get_gmm(
        torch.randn(N_PARTICLES, HORIZON, env_model.action_space.dim),
        torch.ones(N_PARTICLES),
        PRIOR_SIGMA ** 2 * torch.eye(CTRL_DIM),
    )
    init_policies = policies_prior.sample([N_PARTICLES])

    # ... and the model parameters distribution
    loc = torch.as_tensor(PARAMS_LOC)
    mix = dist.Categorical(torch.ones(loc.size(0)))
    comp = dist.Independent(
        dist.Normal(loc, torch.ones_like(loc) * (PARAMS_SIGMA ** 2),),
        reinterpreted_batch_ndims=1,
    )
    dynamics_prior = dist.MixtureSameFamily(mix, comp)
    dynamics_prior = dist.Independent(
        dist.Uniform(torch.tensor([0.6, 0.6]), torch.tensor([1.3, 1.3])), 1
    )

    # Controller hyperparameters
    controller_kwargs = {
        "observation_space": env_model.observation_space,
        "action_space": env_model.action_space,
        "hz_len": HORIZON,  # control horizon
        "action_samples": ACTION_SAMPLES,  # sampled trajectories
        "params_samples": PARAMS_SAMPLES,  # sampled params
        "temperature": 1 / ALPHA,  # temperature
        "a_cov": CTRL_SIGMA ** 2 * torch.eye(CTRL_DIM),
        "inst_cost_fn": inst_cost,
        "term_cost_fn": term_cost,
    }

    # SVMPC
    kernel_type = exp_params["kernel"]
    if kernel_type == "message_passing":
        base_kernel = RBF(bandwidth=-1)
        kernel = iid_mp(base_kernel=base_kernel, ctrl_dim=1, indep_controls=True)
    elif kernel_type == "rbf":
        kernel = RBFKernel()
    else:
        raise ValueError("Kernel type '{}' is not valid.".format(kernel_type))

    likelihood_kwargs = {"alpha": ALPHA, "n_samples": ACTION_SAMPLES}
    svmpc_kwargs = {
        "init_particles": init_policies,
        "prior": policies_prior,
        "kernel": kernel,
        "n_particles": N_PARTICLES,
        "bw_scale": BANDWIDTH_SCALE,
        "n_steps": 1,
        "optimizer_class": torch.optim.SGD,
        "lr": LEARNING_RATE,
    }

    mpf_n_part = exp_params["mpf_n_particles"]
    mpf_steps = exp_params["mpf_steps"]
    mpf_log_space = exp_params["mpf_log_space"]
    mpf_bw = exp_params["mpf_bandwidth"]
    mpf_init = dynamics_prior.sample([mpf_n_part])
    if mpf_log_space:
        mpf_init = mpf_init.clamp(min=1e-6).log()
    dynamics_lik = GaussianLikelihood(
        initial_obs=init_state,
        obs_std=exp_params["mpf_obs_std"],
        model=PendulumModel(uncertain_params=("length", "mass")),
        log_space=exp_params["mpf_log_space"],
    )
    mpf = MPF(
        init_particles=mpf_init,
        likelihood=dynamics_lik,
        optimizer_class=torch.optim.SGD,
        lr=exp_params["mpf_learning_rate"],
        bw=mpf_bw,
        bw_scale=exp_params["mpf_bandwidth_scaling"],
    )

    # UT hyperparameters
    tf = MerweScaledUTF(
        n=config_data["utf"]["n"], alpha=config_data["utf"]["alpha"]
    )  # number of sigma points and scaling

    # Create a list of randomly sampled true parameters, one per episode. All
    # experiments use the same list.
    parameters_set = [
        {"length": item[0], "mass": item[1]}
        for item in dynamics_prior.sample([sim_kwargs["episodes"]])
    ]

    result_df = pd.DataFrame([])
    save_path = save_progress(params=config_data)

    # =====================================
    # ========== EXPERIMENT LOOP ==========
    # =====================================

    # ========== DUAL SVMPC ==========
    case = "DuSt-MPC"
    print("\nRunning {} simulation:".format(case))
    model_kwargs = {
        "uncertain_params": ("length", "mass"),
    }
    controller = MultiDISCO(
        params_sampling=True,
        n_policies=N_PARTICLES,
        params_log_space=exp_params["mpf_log_space"],
        **controller_kwargs,
    )
    dual_df = run_pendulum_simulation(
        init_state=init_state,
        init_policies=init_policies,
        model_kwargs=model_kwargs,
        dyn_dist=dynamics_prior,
        experiment_params=parameters_set,
        controller=controller,
        use_exact_model=False,
        use_svmpc=True,
        svmpc_kwargs=svmpc_kwargs,
        lik_kwargs=likelihood_kwargs,
        mpf=mpf,
        mpf_bw=mpf_bw,
        **sim_kwargs,
    )
    dual_df["Case"] = case
    result_df = pd.concat((result_df, dual_df), axis=0,)

    # ========== SVMPC WITH MEAN PARAMS ==========
    case = "SVMPC"
    print("\nRunning {} simulation:".format(case))
    model_kwargs = {
        "uncertain_params": None,
    }
    controller = MultiDISCO(
        params_sampling=None, n_policies=N_PARTICLES, **controller_kwargs,
    )
    svmpc_df = run_pendulum_simulation(
        init_state=init_state,
        init_policies=init_policies,
        model_kwargs=model_kwargs,
        dyn_dist=dynamics_prior,
        experiment_params=parameters_set,
        controller=controller,
        use_exact_model=False,
        use_svmpc=True,
        svmpc_kwargs=svmpc_kwargs,
        lik_kwargs=likelihood_kwargs,
        mpf=None,
        **sim_kwargs,
    )
    svmpc_df["Case"] = case
    result_df = pd.concat((result_df, svmpc_df), axis=0,)

    # ========== BASELINE CASE ==========
    case = "MPPI Baseline"
    print("\nRunning {} simulation:".format(case))
    model_kwargs = {
        "uncertain_params": None,
    }
    controller = MultiDISCO(params_sampling=None, n_policies=1, **controller_kwargs,)
    mppi_base_df = run_pendulum_simulation(
        init_state=init_state,
        init_policies=init_policies[0].unsqueeze(0),
        model_kwargs=model_kwargs,
        dyn_dist=dynamics_prior,
        experiment_params=parameters_set,
        controller=controller,
        use_exact_model=True,
        use_svmpc=False,
        **sim_kwargs,
    )
    mppi_base_df["Case"] = case
    result_df = pd.concat((result_df, mppi_base_df), axis=0,)

    # ========== DISCO WITH SINGLE POLICY ==========
    case = "DISCO"
    print("\nRunning {} simulation:".format(case))
    model_kwargs = {
        "uncertain_params": ("length", "mass"),
    }
    # Single policy controller

    controller = MultiDISCO(
        params_sampling=tf, n_policies=1, params_log_space=False, **controller_kwargs,
    )
    disco_utf_df = run_pendulum_simulation(
        init_state=init_state,
        init_policies=init_policies[0].unsqueeze(0),
        model_kwargs=model_kwargs,
        dyn_dist=dynamics_prior,
        experiment_params=parameters_set,
        controller=controller,
        use_exact_model=False,
        use_svmpc=False,
        **sim_kwargs,
    )
    disco_utf_df["Case"] = case
    result_df = pd.concat((result_df, disco_utf_df), axis=0,)

    save_progress(folder_name=save_path.stem, data=result_df.reset_index().to_dict())
