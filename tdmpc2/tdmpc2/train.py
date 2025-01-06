import os
# 環境変数の設定（MujocoのレンダリングやTorchのログ関連）
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"

import warnings
warnings.filterwarnings('ignore')  # 警告を非表示
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg  # Hydraで読み込んだ設定を解析するための関数
from common.seed import set_seed     # 乱数シードを固定するための関数
from common.buffer import Buffer     # 経験を格納するバッファ
from envs import make_env           # 環境を作成する関数
from tdmpc2 import TDMPC2           # TD-MPC2のエージェントクラス
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger     # ログを管理するクラス

torch.backends.cudnn.benchmark = True  # CNNでの高速化を有効化
torch.set_float32_matmul_precision('high')  # 行列演算の精度を調整


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	シングルタスク / マルチタスクのTD-MPC2エージェントを学習するためのスクリプトです。

	主な引数:
	  - task: タスク名 (マルチタスク学習ではmt30/mt80を使用)
	  - model_size: モデルサイズ (1, 5, 19, 48, 317のいずれか)
	  - steps: 環境とのインタラクション回数 (既定: 10M)
	  - seed: 乱数シード (既定: 1)

	その他の引数はconfig.yamlを参照してください。

	使用例:
	  python train.py task=mt80 model_size=48
	  python train.py task=mt30 model_size=317
	  python train.py task=dog-run steps=7000000
	"""
	assert torch.cuda.is_available(), "CUDAが利用できません"
	assert cfg.steps > 0, '学習ステップ数は1以上にしてください'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	# マルチタスクかどうかでTrainerクラスを切り替える
	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer

	# Trainerインスタンスを作成し、学習を実行する
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),      # cfgに基づいて環境を生成
		agent=TDMPC2(cfg),      # TD-MPC2エージェントを生成
		buffer=Buffer(cfg),     # 経験を保存するバッファ
		logger=Logger(cfg),     # 学習ログを記録するロガー
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
