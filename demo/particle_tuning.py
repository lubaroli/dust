from pathlib import Path

import optuna
import torch
import torch.distributions as dist
import yaml
from dust.controllers.disco import MultiDISCO
from dust.inference import likelihoods
from dust.inference.svgd import get_gmm
from dust.inference.svmpc import SVMPC
from dust.kernels.base_kernels import RBF
from dust.kernels.composite_kernels import iid_mp
from dust.models.particle import Particle
from dust.utils.helper import save_progress
from dust.utils.simulations import run_particle_episode
from gpytorch.kernels.rbf_kernel import RBFKernel


def objective(trial):
    prior_sigma = PRIOR_SIGMA
    prior_weighted = WEIGHTED_PRIOR
    ctrl_sigma = CTRL_SIGMA
    horizon = HORIZON
    lr = LEARNING_RATE
    alpha = ALPHA
    prior_sigma = PRIOR_SIGMA
    if USE_SVMPC:
        # Hyperparameters to tune
        lr = trial.suggest_loguniform("lr", 0.1, 100)
        alpha = trial.suggest_loguniform("alpha", 1e-1, 1e1)
        prior_sigma = trial.suggest_loguniform("prior_sigma", 1, 100)
        prior_weighted = bool(trial.suggest_int("prior_weighted", 0, 1))
        ctrl_sigma = trial.suggest_uniform("ctrl_sigma", 1, 100)
        horizon = trial.suggest_int("horizon", 10, 30)

    # ========== EXPERIMENT LOOP ==========
    policies_prior = get_gmm(
        torch.randn(N_PARTICLES, horizon, CTRL_DIM),
        torch.ones(N_PARTICLES),
        prior_sigma ** 2 * torch.eye(CTRL_DIM),
    )
    init_policies = policies_prior.sample([N_PARTICLES])

    dynamics_prior = getattr(dist, DYN_PRIOR)(DYN_PRIOR_ARG1, DYN_PRIOR_ARG2)
    system_kwargs = {
        "uncertain_params": ["mass"],
        "mass": dynamics_prior.mean,
    }
    # Model is used for the internal rollouts, system is the simulator,
    # they may have different parameters.
    base_model = Particle(**env_params, **system_kwargs)

    controller = MultiDISCO(
        base_model.observation_space,
        base_model.action_space,
        horizon,
        N_PARTICLES,
        ACTION_SAMPLES,
        temperature=1 / alpha,
        a_cov=ctrl_sigma ** 2 * torch.eye(CTRL_DIM),
        ctrl_penalty=1.0,
        params_sampling=SAMPLING,
        inst_cost_fn=base_model.default_inst_cost,
        term_cost_fn=base_model.default_term_cost,
        init_actions=init_policies,
    )

    kernel_type = exp_params["kernel"]
    if kernel_type == "message_passing":
        base_kernel = RBF(bandwidth=-1)
        kernel = iid_mp(base_kernel=base_kernel, ctrl_dim=2, indep_controls=True)
    elif kernel_type == "rbf":
        kernel = RBFKernel()
    else:
        raise ValueError("Kernel type '{}' is not valid.".format(kernel_type))
    likelihood_pol = getattr(likelihoods, LIKELIHOOD)(
        alpha, controller=controller, model=base_model, n_samples=ACTION_SAMPLES
    )
    svmpc_kwargs = {
        "init_particles": init_policies.detach().clone(),
        "prior": policies_prior,
        "likelihood": likelihood_pol,
        "kernel": kernel,
        "n_particles": N_PARTICLES,
        "bw_scale": BANDWIDTH_SCALE,
        "n_steps": 1,
        "optimizer_class": torch.optim.SGD,
        "lr": lr,
        "weighted_prior": prior_weighted,
    }
    svmpc = SVMPC(**svmpc_kwargs)
    # Try/except to avoid runtime errors
    try:
        loss = run_particle_episode(
            init_state=INIT_STATE,
            model=base_model,
            dyn_dist=dynamics_prior,
            controller=controller,
            use_svmpc=True,
            svmpc=svmpc,
            load=0,
        )
    except RuntimeError:
        loss = float("inf")
    return loss


if __name__ == "__main__":
    config_file = Path("demo/particle_config.yaml")
    with config_file.open() as f:
        config_data = yaml.load(f, yaml.FullLoader)

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

    # ========== EXPERIMENT HYPERPARAMETERS ==========
    INIT_STATE = torch.as_tensor(env_params["init_state"]).clone()
    HORIZON = exp_params["horizon"]
    N_PARTICLES = exp_params["n_particles"]
    ACTION_SAMPLES = exp_params["action_samples"]
    PARAMS_SAMPLES = exp_params["params_samples"]
    ALPHA = exp_params["alpha"]
    LEARNING_RATE = exp_params["learning_rate"]
    BANDWIDTH_SCALE = exp_params["bandwidth_scaling"]
    CTRL_SIGMA = exp_params["ctrl_sigma"]
    CTRL_DIM = exp_params["ctrl_dim"]
    WARM_UP = exp_params["warm_up"]
    LIKELIHOOD = exp_params["likelihood"]
    STEPS = exp_params["steps"]
    USE_SVMPC = exp_params["use_svmpc"]
    PRIOR_SIGMA = exp_params["prior_sigma"]
    WEIGHTED_PRIOR = exp_params["weighted_prior"]
    DYN_PRIOR = exp_params["dyn_prior"]
    DYN_PRIOR_ARG1 = exp_params["dyn_prior_arg1"]
    DYN_PRIOR_ARG2 = exp_params["dyn_prior_arg2"]
    LOAD = exp_params["extra_load"]
    SAMPLING = exp_params["sampling"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500)

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
