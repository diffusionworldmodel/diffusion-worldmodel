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
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
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
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
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
			for key in ["encoder", "dynamics", "reward", "pi"]:
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
	def _estimate_value(self, z, actions, task):
		"""潜在状態zから始まり、指定されたアクションを実行する軌道の価値を推定します。"""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		action, _ = self.model.pi(z, task)
		return G + discount * self.model.Q(z, action, task, return_type='avg')

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
		# ポリシー軌道をサンプル
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		# 状態とパラメータの初期化
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# MPPIの反復
		for _ in range(self.cfg.iterations):

			# アクションをサンプル
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# エリートアクションを計算
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# パラメータを更新
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# アクションを選択
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))  # gumbel_softmax_sampleはcudaグラフと互換性あり
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
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

	def _update(self, obs, action, reward, task=None):
		# ターゲットを計算
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)

		# 更新の準備
		self.model.train()

		# 潜在ロールアウト
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# 予測
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)

		# 損失を計算
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
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

		# トレーニング統計を返す
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
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
