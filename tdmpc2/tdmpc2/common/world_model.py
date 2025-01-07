from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, math, init, temporal, diffusion
from tensordict import TensorDict
from tensordict.nn import TensorDictParams
from dataclasses import dataclass

@dataclass
class Config:
    # misc
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = '/home/aajay/weights/'
    dataset = 'hopper-medium-expert-v2'

    ## model
    n_diffusion_steps = 100 # 200
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    calc_energy=False
    dim=128
    condition_dropout=0.25
    condition_guidance_w = 1.2
    test_ret=0.9
    renderer = 'utils.MuJoCoRenderer'

    ## dataset
    loader = 'datasets.SequenceDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.99
    max_path_length = 1000
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    termination_penalty = -100
    returns_scale = 400.0 # Determined using rewards from the dataset

    ## training
    n_steps_per_epoch = 10000
    loss_type = 'l2'
    n_train_steps = 1e6
    batch_size = 32
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 10000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = False

class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		# Encoder
		enc_dict = {}
		for k in cfg.obs_shape.keys():
			if k == 'state':
				enc_dict[k] = layers.MLP(
					input_dim=cfg.obs_shape[k][0] + cfg.task_dim,
					hidden_dims=max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim],
					output_dim=cfg.latent_dim,
					activation=layers.SimNorm(cfg)
				)
			elif k == 'rgb':
				enc_dict[k] = layers.Encoder(cfg)
			else:
				raise ValueError(f"Unknown observation type: {k}")
		self._encoder = nn.ModuleDict(enc_dict)
		# DynamicsDiffusion設定
		unet_model = temporal.TemporalUnet(
			horizon=cfg.horizon, # Config.horizon,
			transition_dim=cfg.latent_dim + cfg.action_dim,
			cond_dim=cfg.latent_dim,
			dim_mults=Config.dim_mults,
			returns_condition=Config.returns_condition,
			dim=Config.dim,
			condition_dropout=Config.condition_dropout,
			calc_energy=Config.calc_energy,
		)
		self.dynamics_diffusion = diffusion.GaussianDiffusion(
			model=unet_model,
			horizon=cfg.horizon, # Config.horizon,
			observation_dim=cfg.latent_dim,
			action_dim=cfg.action_dim,
			n_timesteps=Config.n_diffusion_steps,
			loss_type=Config.loss_type,
			clip_denoised=Config.clip_denoised,
			predict_epsilon=Config.predict_epsilon,
			## loss weighting
			action_weight=Config.action_weight,
			loss_weights=Config.loss_weights,
			loss_discount=Config.loss_discount,
			returns_condition=Config.returns_condition,
			condition_guidance_w=Config.condition_guidance_w,
		)
		# Reward, policy prior, Q-functions
		self._reward = layers.MLP(
			input_dim=cfg.latent_dim + cfg.action_dim + cfg.task_dim,
			hidden_dims=[cfg.mlp_dim, cfg.mlp_dim],
			output_dim=max(cfg.num_bins, 1)
		)
		self._pi = layers.MLP(
			input_dim=cfg.latent_dim + cfg.task_dim,
			hidden_dims=[cfg.mlp_dim, cfg.mlp_dim],
			output_dim=2*cfg.action_dim
		)
		self._Qs = layers.Ensemble([
			layers.MLP(
				input_dim=cfg.latent_dim + cfg.action_dim + cfg.task_dim,
				hidden_dims=[cfg.mlp_dim, cfg.mlp_dim],
				output_dim=max(cfg.num_bins, 1),
				dropout=cfg.dropout
			)
			for _ in range(cfg.num_q)
		])

		self.apply(init.weight_init)
		init.zero_([self._reward.net[-1].weight, self._Qs.params[1]['net']['0']['ln']['weight'], self._Qs.params[1]['net']['1']['ln']['weight']])
		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		self._detach_Qs.params = self._detach_Qs_params
		self._target_Qs.params = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Reward', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._reward, self._pi, self._Qs]):
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def generate_trajectory(self, z, num_samples):
		"""
		zとaの結合から軌道を生成する。
		return actions, latent_states
		return [num_samples, horizon, action_dim], [num_samples, horizon, latent_dim]
		"""
		if self.dynamics_diffusion.returns_condition:
			returns = torch.ones(num_samples, 1).to(z.device)
		else:
			returns = None
		conditions = {0: z}
		if self.dynamics_diffusion.model.calc_energy:
			samples = self.dynamics_diffusion.grad_conditional_sample(conditions, returns=returns)
		else: # こっち
			samples = self.dynamics_diffusion.conditional_sample(conditions, returns=returns)
		return samples[:, :, :self.cfg.action_dim], samples[:, :, self.cfg.action_dim:]

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"entropy": -log_prob,
			"entropy_scale": self.cfg.entropy_coef * scaled_log_prob / log_prob,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
