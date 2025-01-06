import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from tensordict import TensorDict

class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2エージェント。トレーニングと推論を実装。
	単一タスクとマルチタスクの両方の実験に使用でき、
	状態とピクセルの観察の両方をサポートします。
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
		], lr=self.cfg.lr, capturable=True)
		self.optimizer = torch.optim.Adam(self.model.dynamics_diffusion.parameters(), lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # 大きなアクション空間に対するヒューリスティック
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('更新関数をtorch.compileでコンパイル中...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None) # キャッシュされた計画関数
		if _plan_val is not None:
			return _plan_val # キャッシュされた計画関数を返す
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		エピソードの長さに応じた割引率を返します。
		エピソードの長さに応じて割引率を線形にスケールする簡単なヒューリスティック。
		デフォルト値はほとんどのタスクでうまく機能するはずですが、必要に応じて変更できます。

		Args:
			episode_length (int): エピソードの長さ。エピソードは固定長であると仮定します。

		Returns:
			float: タスクの割引率。
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		エージェントの状態辞書をファイルパスに保存します。

		Args:
			fp (str): 状態辞書を保存するファイルパス。
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		保存された状態辞書をファイルパス（または辞書）から現在のエージェントにロードします。

		Args:
			fp (str or dict): ロードするファイルパスまたは状態辞書。
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		try: # 2023年11月10日以降に作成されたチェックポイント
			self.model.load_state_dict(state_dict)
		except: # 後方互換性
			def _get_submodule(state_dict, key):
				return {k.replace(f"_{key}.", ""): v for k, v in state_dict.items() if k.startswith(f"_{key}.")}
			for key in ["encoder", "reward", "pi"]:
				submodule_state_dict = _get_submodule(state_dict, key)
				getattr(self.model, f"_{key}").load_state_dict(submodule_state_dict)
			# Q関数は特別な処理が必要
			Qs_state_dict = _get_submodule(state_dict, "Qs")
			# TODO: 古いチェックポイントからQ関数の状態辞書をロードする方法を考える
			raise NotImplementedError("古いチェックポイントのロードに関する後方互換性は現在壊れています。" \
							 "修正が発行されるまで、以前のチェックポイントに戻してください。例: 88095e7899497cf7a1da36fb6bbb6bc7b5370d53")
		self.model.load_state_dict(state_dict)

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		ワールドモデルの潜在空間で計画を立ててアクションを選択します。

		Args:
			obs (torch.Tensor): 環境からの観察。
			t0 (bool): エピソードの最初の観察かどうか。
			eval_mode (bool): アクション分布の平均を使用するかどうか。
			task (int): タスクインデックス（マルチタスク実験でのみ使用）。

		Returns:
			torch.Tensor: 環境で取るアクション。
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, latents, actions, task):
		"""潜在状態zから始まり、指定されたアクションを実行する軌道の価値を推定します。"""
		G, discount = 0, 1
		# actions [num_samples, horizon, action_dim] -> [num_samples*horizon, action_dim]
		actions_ = actions.view(-1, self.cfg.action_dim)
		# latents [num_samples, horizon, latent_dim] -> [num_samples*horizon, latent_dim]
		latents_ = latents.view(-1, self.cfg.latent_dim)
		rewards = self.model.reward(latents_, actions_, task)
		# rewards [num_samples*horizon] -> [num_samples, horizon]
		rewards = rewards.view(latents.size(0), -1)
		for t in range(self.cfg.horizon):
			G = G + discount * rewards[:, t]
			discount_update = self.discount[task] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		# G [num_samples]-> [num_samples, 1]
		G = G.unsqueeze(1)
		Q_value = self.model.Q(latents[:,-1], actions[:,-1], task, return_type='avg')
		return G + discount * Q_value # [num_samples, 1]

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		学習されたワールドモデルを使用してアクションのシーケンスを計画します。

		Args:
			z (torch.Tensor): 計画を立てる潜在状態。
			t0 (bool): エピソードの最初の観察かどうか。
			eval_mode (bool): アクション分布の平均を使用するかどうか。
			task (Torch.Tensor): タスクインデックス（マルチタスク実験でのみ使用）。

		Returns:
			torch.Tensor: 環境で取るアクション。
		"""
		z = self.model.encode(obs, task)
		z = z.repeat(self.cfg.num_samples, 1) # [num_samples, latent_dim]
		actions, latents = self.model.generate_trajectory(z, self.cfg.num_samples)
		value = self._estimate_value(latents, actions, task)
		elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
		elite_value, elite_actions = value[elite_idxs], actions[elite_idxs]
		elite_actions = elite_actions.transpose(0, 1)
		max_value = elite_value.max(0).values
		score = torch.exp(self.cfg.temperature*(elite_value - max_value)) # softmax
		score = score / score.sum(0) # 正規化
		mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9) # 重み付き平均
		std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt() # 重み付き標準偏差
		std = std.clamp(self.cfg.min_std, self.cfg.max_std)
		idx = torch.randint(0, self.cfg.num_elites, (1,))
		a = elite_actions[0,idx]
		std = std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		潜在状態のシーケンスを使用してポリシーを更新します。

		Args:
			zs (torch.Tensor): 潜在状態のシーケンス。
			task (torch.Tensor): タスクインデックス（マルチタスク実験でのみ使用）。

		Returns:
			float: ポリシー更新の損失。
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# 損失はQ値の重み付き和
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(info["entropy_scale"] * info["entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_entropy_scale": info["entropy_scale"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		報酬と次のタイムステップの観察からTDターゲットを計算します。

		Args:
			next_z (torch.Tensor): 次のタイムステップの潜在状態。
			reward (torch.Tensor): 現在のタイムステップの報酬。
			task (torch.Tensor): タスクインデックス（マルチタスク実験でのみ使用）。

		Returns:
			torch.Tensor: TDターゲット。
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(self, obs, actions, reward, task=None):
		# ターゲットを計算
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)

		# 更新の準備
		self.model.train()

		zs = torch.empty(self.cfg.batch_size, self.cfg.horizon, self.cfg.latent_dim, device=self.device)
		# obs [horizon+1, batch_size,obs_dim]
		obs = obs.view(self.cfg.batch_size*(self.cfg.horizon+1), -1)
		z_list = self.model.encode(obs, task)
		z_list = z_list.view((self.cfg.horizon+1), self.cfg.batch_size, self.cfg.latent_dim)
		z = z_list[0] # [batch_size, latent_dim]
		_, zs = self.model.generate_trajectory(z, self.cfg.batch_size)

		zs = zs.transpose(0, 1) # [horizon, batch_size, latent_dim]
		# actions [horizon, batch_size, action_dim]
		print("zs shape: ", zs.shape) # [16, 256, 512]
		print("actions shape: ", actions.shape) # [16, 256, 6]
		# 予測
		qs = self.model.Q(zs, actions, task, return_type='all')
		print("qs shape: ", qs.shape) # [5, 16, 256, 101]
		reward_preds = self.model.reward(zs, actions, task)
		print("reward_preds shape: ", reward_preds.shape) # [16, 256, 101]

		# 損失を計算
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		reward_loss = reward_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# モデルを更新
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# ポリシーを更新
		pi_info = self.update_pi(zs.detach(), task)

		# ターゲットQ関数を更新
		self.model.soft_update_target_Q()

		# diffusionの更新
		conditions = {0: z}
		returns = torch.ones(self.cfg.batch_size, 1).to(z.device)
		az = torch.cat([actions, z_list[:-1]], dim=-1)
		# az [horizon, batch_size, action_dim+latent_dim]->[batch_size, horizon, action_dim+latent_dim]
		az = az.transpose(0, 1)
		# z [batch_size, latent_dim]
		diffusion_loss, _ = self.model.dynamics_diffusion.loss(az, conditions, returns)
		diffusion_loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()

		# トレーニング統計を返す
		self.model.eval()
		info = TensorDict({
			"diffusion_loss": diffusion_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		メインの更新関数。モデル学習の1回の反復に対応します。

		Args:
			buffer (common.buffer.Buffer): リプレイバッファ。

		Returns:
			dict: トレーニング統計の辞書。
		"""
		obs, action, reward, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, **kwargs)
