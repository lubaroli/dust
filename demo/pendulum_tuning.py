import math
from pathlib import Path

import optuna
import torch
import torch.distributions as dist
import yaml
from dust.controllers.disco import MultiDISCO
from dust.inference.svgd import get_gmm
from dust.kernels.composite_kernels import iid_mp
from dust.models.pendulum import PendulumModel
from dust.utils.helper import save_progress
from dust.utils.simulations import run_pendulum_simulation
from dust.utils.utf import MerweScaledUTF
from gpytorch.kernels import RBFKernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inst_cost(states, controls=None, n_pol=1, debug=None):
    # Note that theta may range beyond 2*pi
    theta, theta_d = states.chunk(2, dim=1)
    return 50.0 * (theta.cos() - 1) ** 2 + 1.0 * theta_d ** 2


def term_cost(states, n_pol=1, debug=None):
    return inst_cost(states).squeeze()


def objective(trial):
    if USE_SVMPC:
        # Hyperparameters to tune
        lr = trial.suggest_loguniform("lr", 1e-1, 1e2)
        alpha = trial.suggest_uniform("alpha", 1e-1, 1e1)
        prior_sigma = trial.suggest_loguniform("prior_sigma", 1e-1, 1e2)
        horizon = trial.suggest_int("horizon", 10, 30)
    else:
        # Hyperparameters to tune
        alpha = trial.suggest_uniform("alpha", 1e-1, 1e1)
        horizon = trial.suggest_int("horizon", 10, 30)
        # and dummy ones
        lr = 1.0
        prior_sigma = 1.0

    # Define environment...
    env_model = PendulumModel()
    sim_kwargs = {
        "episodes": 1,
        "render": False,
        "warm_up": 0,
    }

    # Policies prior...
    policies_prior = get_gmm(
        torch.randn(N_PARTICLES, horizon, env_model.action_space.dim),
        torch.ones(N_PARTICLES),
        prior_sigma ** 2 * torch.eye(CTRL_DIM),
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

    # Create a list of randomly sampled true parameters, one per iteration. All
    # experiments use the same list.
    parameters_set = [
        {"length": item[0], "mass": item[1]} for item in dynamics_prior.sample([1])
    ]
    # Controller hyperparameters
    controller_kwargs = {
        "observation_space": env_model.observation_space,
        "action_space": env_model.action_space,
        "hz_len": horizon,  # control horizon
        "action_samples": ACTION_SAMPLES,  # sampled trajectories
        "params_samples": 0,  # sampled params
        "params_log_space": exp_params["mpf_log_space"],
        "temperature": 1 / alpha,  # temperature
        "a_cov": CTRL_SIGMA ** 2 * torch.eye(CTRL_DIM),
        "inst_cost_fn": inst_cost,
        "term_cost_fn": term_cost,
    }

    # Stein optimizer
    kernel_type = exp_params["kernel"]
    if kernel_type == "message_passing":
        base_kernel = RBF(bandwidth=-1)
        kernel = iid_mp(base_kernel=base_kernel, ctrl_dim=1, indep_controls=True)
    elif kernel_type == "rbf":
        kernel = RBFKernel()
    else:
        raise ValueError("Kernel type '{}' is not valid.".format(kernel_type))

    likelihood_kwargs = {"alpha": alpha, "n_samples": ACTION_SAMPLES}
    svmpc_kwargs = {
        "init_particles": init_policies,
        "prior": policies_prior,
        "kernel": kernel,
        "n_particles": N_PARTICLES,
        "bw_scale": BANDWIDTH_SCALE,
        "n_steps": 1,
        "optimizer_class": torch.optim.SGD,
        "lr": lr,
    }

    # UT hyperparameters
    tf = MerweScaledUTF(
        n=config_data["utf"]["n"], alpha=config_data["utf"]["alpha"]
    )  # number of sigma points and scaling
    if USE_UTF:
        sampling = tf
    else:
        sampling = SAMPLING

    model_kwargs = {
        "uncertain_params": ("length", "mass"),
    }

    # Single policy controller
    controller = MultiDISCO(
        params_sampling=sampling, n_policies=N_PARTICLES, **controller_kwargs
    )
    res_df = run_pendulum_simulation(
        init_state=INIT_STATE,
        init_policies=init_policies,
        model_kwargs=model_kwargs,
        dyn_dist=dynamics_prior,
        experiment_params=parameters_set,
        controller=controller,
        use_exact_model=EXACT_MODEL,
        use_svmpc=USE_SVMPC,
        svmpc_kwargs=svmpc_kwargs,
        lik_kwargs=likelihood_kwargs,
        mpf=None,
        **sim_kwargs,
    )
    return float(res_df["Cost"].mean())


if __name__ == "__main__":
    config_file = Path("demo/pendulum_config.yaml")
    with config_file.open() as f:
        config_data = yaml.load(f, yaml.FullLoader)

    exp_params = config_data["exp_params"]

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
    INIT_STATE = torch.as_tensor(exp_params["init_state"]).clone()

    # Setup trial
    USE_UTF = False
    SAMPLING = False
    USE_SVMPC = True
    EXACT_MODEL = True
    N_PARTICLES = 3

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    save_progress(data=study)
