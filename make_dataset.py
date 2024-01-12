
# This script prepares the training images and ratings for the training.
# It assumes that all images are stored as files that PIL can read.
# It also assumes that the paths to the images files and the average ratings are in a .parquet files that can be read into a dataframe ( df ).
# このスクリプトは、トレーニング用のトレーニング画像と評価を準備します。
# すべての画像が PIL が読み取れるファイルとして保存されていることを前提としています。
# また、画像ファイルへのパスと平均評価が、データフレーム ( df ) に読み込める .parquet ファイル内にあることも前提としています。

import open_clip
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import time
import os
import glob
import tqdm
import yaml

# スクリプトのあるディレクトリのパスを取得
script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# YAMLファイルのパスを組み立て
config_path = os.path.join(script_dir, "conf", "customscore.yaml")
# YAMLファイルの読み込み
with open(config_path, "r", encoding="utf-8") as file:
   config = yaml.safe_load(file)


def load_open_clip_model(model_name="ViT-H-14"):
   """モデルをロードする関数
   YAMLにカスタムモデルが設定されている場合カスタムモデルをロードする。
   Args:
      model_name (str): モデルの名前 (デフォルト: "ViT-H-14")

   Returns:
      model: ロードされたOpen CLIPモデル
      preprocess: データの前処理を行うための関数
   """
   device = torch.device(config["device"])
   custom_model_path = config["custom_model"]

   if custom_model_path:
      custom_model = torch.load(custom_model_path, map_location=device)
      #TODO: カスタムモデルのベースモデル名を取得する方法を考える
      # 指定されたモデル名でOpen CLIPモデルをロード

   model, preprocess_train, preprocess = open_clip.create_model_and_transforms(model_name, device=device)
   return model, preprocess, custom_model


def normalized(feature_vector , axis=-1, order=2):
   """
   特徴ベクトルの正則化（Normalization of Feature Vector）
   この関数 normalized は、与えられた特徴ベクトルを正則化するために使われるんだぜ。
   正則化とは、特徴ベクトルの各要素をスケーリングして、全体の大きさ（ノルム）が1になるようにすることだ。
   これによって、異なるデータセットやモデル間で特徴ベクトルを比較しやすくなるぜ。

   パラメーター:
   feature_vector (ndarray): 正規化する対象のNumPy配列（特徴量ベクトル）
   axis (int, optional): 正規化する軸の番号。デフォルトは-1です。
   order (int, optional): 正規化の順序。デフォルトは2です。

   **処理の流れ**:
   まず、`np.linalg.norm` を使って、指定された軸に沿ったノルムを計算するぜ。
   `order` パラメータでノルムの種類を指定できる。
   デフォルトではユークリッドノルムを計算する。
   次に、計算されたノルムが0にならないようにしてる。ノルムが0の場合、除算でエラーが発生するからな。
   最後に、元の特徴ベクトルをノルムで割って、正則化したベクトルを得る。
   """
   l2_norm = np.atleast_1d(np.linalg.norm(feature_vector, order, axis))
   l2_norm[l2_norm == 0] = 1
   return feature_vector / np.expand_dims(l2_norm, axis)

if __name__ == "__main__":
   class CustomModel(nn.Module):
      def __init__(self, base_model):
         super(CustomModel, self).__init__()
         # 基本モデルの特定の層をカスタムモデルにコピー
         self.conv1 = base_model.visual.conv1
         if hasattr(base_model.visual, "bn1"):
            self.bn1 = base_model.visual.bn1
         self.relu = nn.ReLU(inplace=True)
         if hasattr(base_model.visual, "maxpool"):
            self.maxpool = base_model.visual.maxpool

      def forward(self, x):
         x = self.conv1(x)
         x = self.bn1(x)
         x = self.relu(x)
         x = self.maxpool(x)
         # 他のレイヤーの順伝播もここに追加してください
         return x


   # CLIPモデルをロード
   model, preprocess, custom_model = load_open_clip_model()

   # 独自の学習済みモデルの畳み込み層をCLIPモデルの対応する層にコピー
   if custom_model:
      custom_model_instance = CustomModel(model)
      state_dict = {f'conv{i // 2}.weight' if i % 2 == 0 else f'conv{i // 2}.bias': v for i, (k, v) in enumerate(custom_model.items())}
      custom_model_instance.load_state_dict(state_dict)
      model.visual.conv1.weight = custom_model_instance.conv1.weight
      model.visual.conv1.bias = custom_model_instance.conv1.bias


   # データディレクトリの設定
   data_dirctory = config["data"]
   dir_list = glob.glob(data_dirctory+"/*")

   img_features = []
   manual_scores = []
   c = 0
   bucket_log = ""

   # データの復元が指定されている場合
   resume_file = config["resume"]
   if resume_file != "":
      img_features = np.load(f"img_features_{resume_file}.npy")
      manual_scores = np.load(f"manual_scores_{resume_file}.npy")
      c = img_features.shape[0]

   start = time.time()

   # 各ディレクトリを処理
   for dir in dir_list:
      score = float(os.path.split(dir)[-1])
      files = glob.glob(dir+"/*.png") + glob.glob(dir+"/*.jpg") + glob.glob(dir+"/*.jpeg")

      if len(files) == 0:
         print(f"{dir} 内に画像がありません")
         continue
      else:
         print(f"{dir} 内の画像読み込み中...")
      imgnum = 0

      # 各画像ファイルを処理
      for f in tqdm.tqdm(files):
         try:
            device = torch.device(config["device"])
            image = preprocess(Image.open(f)).unsqueeze(0).to(device)
         except:
            continue
         # 画像の特徴量を抽出
         with torch.no_grad():
            image_features = model.encode_image(image)

         # 抽出された特徴量を配列に追加
         im_emb_arr = image_features.cpu().detach().numpy()
         img_features.append(normalized(im_emb_arr))

         # スコアを配列に追加
         manual_scores_ = np.zeros((1, 1))
         manual_scores_[0][0] = score
         manual_scores.append(manual_scores_)

         # 画像数とカウントを更新
         imgnum = imgnum + 1
         c += 1

   # 各スコアの画像数をログに追加
   bucket_log += f"score {score} nums: {imgnum}\n"

   #処理結果表示
   print(f"image score nums\n{bucket_log}")
   img_features = np.vstack(img_features)
   manual_scores = np.vstack(manual_scores)
   print(img_features.shape)
   print(manual_scores.shape)

   #データの保存
   output_name = config["output"]
   np.save(f'img_features_{output_name}.npy', img_features)
   np.save(f'manual_scores_{output_name}.npy', manual_scores)