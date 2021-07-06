from abc import ABC, abstractmethod

import torch
import torch.distributions as dist
from torch.distributions.distribution import Distribution
from torch.functional import Tensor

from ..controllers.base import BaseController
from ..models.base import BaseModel


class GaussianLikelihood(ABC):
    def __init__(
        self,
        initial_obs: Tensor,
        obs_std: float,
        model: BaseModel,
        log_space: bool = False,
    ):
        assert (
            initial_obs.ndim == 1
        ), "Gaussian likelihood needs a single dimensional loc tensor."
        self.dim = initial_obs.shape[0]
        self.sigma = obs_std
        # condition also sets loc
        self.density = self.condition(new_obs=initial_obs, action=None)
        self.model = model
        self.log_space = log_space

    def sample(self, theta: Tensor):
        assert (
            self.past_action is not None
        ), "Previous action is None. Need at least one observation to start sampling."
        if self.log_space:
            params = theta.exp()
        else:
            params = theta
        if str(self.model.__class__) == "<class '__main__.SSModel'>":
            states = self.model.step(
                self.past_obs.view(1, -1), self.past_action.view(1, -1), params
            )
        else:
            params_dict = self.model.params_to_dict(params)
            states = self.past_obs.repeat(theta.shape[0], 1)
            states = self.model.step(states, self.past_action, params_dict)
        return states

    def log_prob(self, samples: Tensor):
        return self.density.log_prob(samples).unsqueeze(-1)

    def condition(
        self, action: Tensor, new_obs: Tensor, covariance_matrix: Tensor = None
    ):
        try:
            self.past_obs = self.loc
        except AttributeError:
            self.past_obs = None
        self.loc = new_obs
        self.past_action = action
        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
        self.density = dist.MultivariateNormal(
            self.loc, self.sigma ** 2 * torch.eye(self.dim)
        )


class CostLikelihood(ABC):
    def __init__(
        self, n_samples: int, controller: BaseController, model: BaseModel,
    ):
        self.n_samples = n_samples
        self.last_states = None
        self.last_actions = None
        self.last_policies = None
        self.last_costs = None
        self.params = None
        self.params_log_p = None
        self.controller = controller
        self.model = model

    def sample(self, theta: Tensor, state: Tensor, params_dist: Distribution):
        """
        Generates roll-outs and compute costs using internal model and controller.
        """
        pi = dist.Independent(
            dist.MultivariateNormal(theta, self.controller.a_dist.covariance_matrix), 1,
        )

        # Use rsample to preserve gradient
        actions = pi.rsample([self.n_samples])
        costs, states, _, _, params_log_p = self.controller.forward(
            state, self.model, params_dist, actions
        )

        self.last_costs = costs
        self.last_states = states
        self.last_actions = actions
        self.last_policies = pi
        self.params_log_p = params_log_p
        # return costs
        return costs, actions

    @abstractmethod
    def log_prob(self, costs: Tensor = None):
        pass


class ExpectedCost(CostLikelihood):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def log_prob(self, costs: Tensor = None):
        # If no values given, use last sampled costs
        if costs is None:
            costs = self.last_costs
        else:
            costs = costs
        return -self.alpha * costs.mean(dim=0)


class ExponentiatedUtility(CostLikelihood):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def log_prob(self, costs: Tensor = None):
        # If no values given, use last sampled costs
        if costs is None:
            costs = self.last_costs
        else:
            costs = costs
        return (-self.alpha * costs).logsumexp(0) - torch.as_tensor(
            costs.size(0), dtype=torch.float
        ).log()
