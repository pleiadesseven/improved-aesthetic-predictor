#import webdataset as wds
from PIL import Image
#import io
#import matplotlib.pyplot as plt
#import os
#import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
#import tqdm

from os.path import join
#from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
import glob
import os
import shutil
import tqdm
import yaml
import open_clip


from PIL import Image, ImageFile

# スクリプトのあるディレクトリのパスを取得
script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# YAMLファイルのパスを組み立て
config_path = os.path.join(script_dir, "conf", "customscore.yaml")
# YAMLファイルの読み込み
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

#####  This script will predict the aesthetic score for this image file:

img_dir = config["img_dir"]#自動分類対象のフォルダ
img_path_list = glob.glob(img_dir+"/*.png") + glob.glob(img_dir+"/*.jpg") + glob.glob(img_dir+"/*.jpeg")
pth_name= config["pth_name"]#読み込むpthファイル名
auto_dir_base = config["auto_dir_path"]#自動分類する時の出力先
auto_dir_flag = config["html"] #Trueの時はhtmlは生成しない
auto_dir_step = config["auto_dir_step"] #フォルダに分類する時のスコアステップ、0.5なら[1.0,1.5,2.0,.....9.5]と0.5刻みでスコアごとに分類する、0.4なら[1.0,5.0,9.0]の三段階
auto_dir_list = []
html_w_num = config["w_num"] #html出力時に横列の数

if auto_dir_flag:
    if not os.path.exists(auto_dir_base):
        os.mkdir(auto_dir_base)
    for i in range(int(10/auto_dir_step)):
        key = 1.0 + (i*auto_dir_step)
        auto_dir_list.append(f"{auto_dir_base}/{key:.1f}")

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(1024)  # CLIP embedding dim is 768 for CLIP ViT B 32

s = torch.load(pth_name)   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to("cuda")
model.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
model2, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", device=device)  #RN50x64 
output_str_tr = ""
output_str_img = ""
output_str_score = ""
i = 0
for img_path in tqdm.tqdm(img_path_list):
    pil_image = Image.open(img_path)

    image = preprocess(pil_image).unsqueeze(0).to(device)



    with torch.no_grad():
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy() )

    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

    #print( f"img : {img_path}")
    #print( f"Aesthetic score predicted by the model: {prediction}")
    score = prediction.data.cpu()[0][0]
    if auto_dir_flag:
        for j, f in enumerate(auto_dir_list):
            if prediction<1.0 + ((j+1)*auto_dir_step) or ((j+1)*auto_dir_step>10.0):
                if not os.path.exists(f):
                    os.mkdir(f)
                score_s = f"{score:02.3f}".replace(".", "_")
                shutil.move(img_path, f"{f}/{score_s}_{os.path.basename(img_path)}")
                break
    else:
        if i%html_w_num==0:
            if i>0:
                output_str_tr += f"<tr>{output_str_img}</tr><tr>{output_str_score}</tr>"
            output_str_img = f"<td><img src=\"{img_path}\"></td>"
            output_str_score = f"<td>{score}</td>"
        else:
            output_str_img += f"<td><img src=\"{img_path}\"></td>"
            output_str_score += f"<td>{score}</td>"
    i += 1

if not auto_dir_flag:
    output_str_tr += f"<tr>{output_str_img}</tr><tr>{output_str_score}</tr>"

    output_str = "<html><head><style>table{text-align:center;}\nimg{width:50%;height:50%;max-widht:240px;max-hight:240px;}</style></head><body><table>"
    output_str += output_str_tr
    output_str += "</table></body></html>"

    output_file = open(f'prev.html', 'w', encoding='utf-8')
    output_file.write(output_str)


