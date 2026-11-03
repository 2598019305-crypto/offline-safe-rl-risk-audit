"""Controlled L1 ablation variants for BCQ-Lag.

A0 = original BCQL (no mask, min aggregation)
A1 = finite-horizon cost target mask + min aggregation
A2 = original target + mean aggregation
A3 = finite-horizon cost target mask + mean aggregation

The frozen `osrl_min/osrl/algorithms/bcql.py` is NOT modified. This subclass only
overrides `cost_critic_loss` and `actor_loss` with the two controlled factors,
and records diagnostics.
"""

import torch

from osrl.algorithms.bcql import BCQL


class BCQLAblation(BCQL):
    def __init__(self, *args, cost_terminal_mask=False, actor_cost_agg="min",
                 cost_target_floor=None, cost_nonneg=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_terminal_mask = bool(cost_terminal_mask)
        self.actor_cost_agg = str(actor_cost_agg)
        self.cost_target_floor = cost_target_floor
        self.cost_nonneg = bool(cost_nonneg)
        if self.cost_nonneg:
            self._patch_nonneg(self.cost_critic)
            self._patch_nonneg(self.cost_critic_old)
        self.last_diag = {}

    @staticmethod
    def _patch_nonneg(module):
        orig = module.predict

        def nonneg_predict(obs, act):
            a, b, l1, l2 = orig(obs, act)
            return (
                a.clamp(min=0.0),
                b.clamp(min=0.0),
                [q.clamp(min=0.0) for q in l1],
                [q.clamp(min=0.0) for q in l2],
            )

        module.predict = nonneg_predict

    def cost_critic_loss(self, observations, next_observations, actions, costs, done):
        _, _, q1_list, q2_list = self.cost_critic.predict(observations, actions)
        with torch.no_grad():
            batch_size = next_observations.shape[0]
            obs_next = torch.repeat_interleave(
                next_observations, self.sample_action_num, 0
            ).to(self.device)
            act_targ_next = self.actor_old(obs_next, self.vae.decode(obs_next))
            q1_targ, q2_targ, _, _ = self.cost_critic_old.predict(obs_next, act_targ_next)
            q_targ = self.lmbda * torch.min(q1_targ, q2_targ) + (
                1.0 - self.lmbda
            ) * torch.max(q1_targ, q2_targ)
            q_targ = q_targ.reshape(batch_size, -1).max(1)[0]
            if self.cost_terminal_mask:
                q_targ = (1.0 - done) * q_targ
            backup = costs + self.gamma * q_targ
            if self.cost_target_floor is not None:
                backup = torch.clamp(backup, min=float(self.cost_target_floor))

        loss_cost_critic = self.cost_critic.loss(backup, q1_list) + self.cost_critic.loss(
            backup, q2_list
        )
        self.cost_critic_optim.zero_grad()
        loss_cost_critic.backward()
        self.cost_critic_optim.step()

        qc1 = torch.stack(q1_list, 0).mean(0)
        qc2 = torch.stack(q2_list, 0).mean(0)
        self.last_diag.update(
            {
                "cost_critic_loss": float(loss_cost_critic.item()),
                "cost_target_mean": float(backup.mean().item()),
                "cost_target_std": float(backup.std().item()),
                "cost_target_p90": float(torch.quantile(backup, 0.9).item()),
                "qc1_pred_mean": float(qc1.mean().item()),
                "qc2_pred_mean": float(qc2.mean().item()),
                "qc_pred_std": float(torch.minimum(qc1, qc2).std().item()),
                "qc_negative_ratio": float((torch.minimum(qc1, qc2) < 0).float().mean().item()),
            }
        )
        return loss_cost_critic, {"loss/cost_critic_loss": loss_cost_critic.item()}

    def actor_loss(self, observations):
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.cost_critic.parameters():
            p.requires_grad = False
        for p in self.vae.parameters():
            p.requires_grad = False

        actions = self.actor(observations, self.vae.decode(observations))
        q1_pi, q2_pi, _, _ = self.critic.predict(observations, actions)
        qc1_pi, qc2_pi, _, _ = self.cost_critic.predict(observations, actions)
        if self.actor_cost_agg == "mean":
            qc_pi = 0.5 * (qc1_pi + qc2_pi)
        else:
            qc_pi = torch.min(qc1_pi, qc2_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        with torch.no_grad():
            multiplier = self.controller.control(qc_pi).detach()
        qc_penalty = ((qc_pi - self.qc_thres) * multiplier).mean()
        loss_actor = -q_pi.mean() + qc_penalty

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        self.last_diag.update(
            {
                "actor_loss": float(loss_actor.item()),
                "actor_reward_q_mean": float(q_pi.mean().item()),
                "actor_cost_q_mean": float(qc_pi.mean().item()),
                "qc_penalty": float(qc_penalty.item()),
                "lagrangian": float(multiplier.item()),
            }
        )

        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.cost_critic.parameters():
            p.requires_grad = True
        for p in self.vae.parameters():
            p.requires_grad = True
        return loss_actor, {
            "loss/actor_loss": loss_actor.item(),
            "loss/qc_penalty": qc_penalty.item(),
            "loss/lagrangian": multiplier.item(),
        }

    def cost_critic_loss_mc(self, observations, actions, mc_target):
        """Cost critic loss with a precomputed Monte-Carlo cost-to-go target."""
        _, _, q1_list, q2_list = self.cost_critic.predict(observations, actions)
        backup = mc_target
        loss_cost_critic = self.cost_critic.loss(backup, q1_list) + self.cost_critic.loss(
            backup, q2_list
        )
        self.cost_critic_optim.zero_grad()
        loss_cost_critic.backward()
        self.cost_critic_optim.step()
        qc1 = torch.stack(q1_list, 0).mean(0)
        qc2 = torch.stack(q2_list, 0).mean(0)
        self.last_diag.update(
            {
                "cost_critic_loss": float(loss_cost_critic.item()),
                "cost_target_mean": float(backup.mean().item()),
                "cost_target_std": float(backup.std().item()),
                "cost_target_p90": float(torch.quantile(backup, 0.9).item()),
                "qc1_pred_mean": float(qc1.mean().item()),
                "qc2_pred_mean": float(qc2.mean().item()),
                "qc_pred_std": float(torch.minimum(qc1, qc2).std().item()),
                "qc_negative_ratio": float((torch.minimum(qc1, qc2) < 0).float().mean().item()),
            }
        )
        return loss_cost_critic, {"loss/cost_critic_loss": loss_cost_critic.item()}
