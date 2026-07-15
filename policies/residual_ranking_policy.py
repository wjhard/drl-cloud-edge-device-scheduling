from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


class ResidualRankingPolicy(MaskableMultiInputActorCriticPolicy):
    """Maskable PPO policy whose logits are HEFT ranks plus learned residual deltas."""

    def __init__(self, *args, delta_scale: float = 1.0, rank_scale: float = 1.0, **kwargs):
        self.delta_scale = float(delta_scale)
        self.rank_scale = float(rank_scale)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        th.nn.init.zeros_(self.action_net.weight)
        th.nn.init.zeros_(self.action_net.bias)

    def _residual_action_logits(self, latent_pi: th.Tensor, obs: dict[str, th.Tensor]) -> th.Tensor:
        if "ready_task_upward_ranks" not in obs:
            raise KeyError("ResidualRankingPolicy requires observation['ready_task_upward_ranks']")

        base_logits = obs["ready_task_upward_ranks"].float()
        if base_logits.ndim == 1:
            base_logits = base_logits.unsqueeze(0)
        delta_logits = th.tanh(self.action_net(latent_pi)) * self.delta_scale
        if base_logits.shape != delta_logits.shape:
            raise ValueError(
                "ready_task_upward_ranks shape must match action logits: "
                f"base={tuple(base_logits.shape)}, delta={tuple(delta_logits.shape)}"
            )
        return self.rank_scale * base_logits + delta_logits

    def _get_action_dist_from_latent_and_obs(
        self,
        latent_pi: th.Tensor,
        obs: dict[str, th.Tensor],
    ) -> MaskableDistribution:
        action_logits = self._residual_action_logits(latent_pi, obs)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(
        self,
        obs: dict[str, th.Tensor],
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent_and_obs(latent_pi, obs)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: dict[str, th.Tensor],
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent_and_obs(latent_pi, obs)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(
        self,
        obs: dict[str, th.Tensor],
        action_masks: np.ndarray | None = None,
    ) -> MaskableDistribution:
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent_and_obs(latent_pi, obs)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution
