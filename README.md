# diffusion-worldmodel
松尾研 世界モデル講座 最終課題 拡散モデルと世界モデル (グループ6)

# TODO
- TDMPC2の動作環境を構築
- 依存ライブラリを可能な限り減らす
- 遷移モデルを変更する

```
python tdmpc2/tdmpc2/train.py
```

# 環境構築
```
conda create -n tdmpc python=3.9
conda activate tdmpc
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
# トラブルシューティング
```
ImportError: ('Unable to load EGL library', "Could not find module 'EGL' (or one of its dependencies). Try using the full path with constructor syntax.", 'EGL', None)
```
PATHに`libEGL.dll`が配置されているフォルダのパスを追加する
```
C:\Program Files\NVIDIA Corporation\NVIDIA app\CEF
```