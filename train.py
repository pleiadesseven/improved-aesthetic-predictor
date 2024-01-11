import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here
import tqdm
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
#from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
import yaml
from make_dataset import load_open_clip_model
import numpy as np
from torchvision import transforms


# スクリプトのあるディレクトリのパスを取得
script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# YAMLファイルのパスを組み立て
config_path = os.path.join(script_dir, "conf", "customscore.yaml")
# YAMLファイルの読み込み
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

#define your neural net here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
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

# load the training data
x_file = config["x_file"]
y_file = config["y_file"]

x = np.load(os.path.join(script_dir, f"img_features_{x_file}.npy"))
y = np.load(os.path.join(script_dir, f"manual_scores_{y_file}.npy"))
batch_size = config["batch_size"]
num_workers = 0
print(f"dataset num: {x.shape[0]}")

val_percentage = config["eval_per"] # 5% of the trainingdata will be used for validation # トレーニングデータの 5% が検証に使用

train_border = int(x.shape[0] * (1 - val_percentage) )
print(f"train data num: {train_border} \t val data num: {x.shape[0]-train_border}")

train_tensor_x = torch.Tensor(x[:train_border]) # transform to torch tensor トーチ テンソルに変換する
train_tensor_y = torch.Tensor(y[:train_border])

train_dataset = TensorDataset(train_tensor_x,train_tensor_y) # create your datset # データセットを作成する
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers) # create your dataloader # データローダーを作成する


val_tensor_x = torch.Tensor(x[train_border:]) # transform to torch tensor トーチ テンソルに変換する
val_tensor_y = torch.Tensor(y[train_border:])


print(train_tensor_x.size())
print(val_tensor_x.size())
print( val_tensor_x.dtype)
print( val_tensor_x[0].dtype)

val_dataset = TensorDataset(val_tensor_x,val_tensor_y) # create your datset
val_loader = DataLoader(val_dataset, batch_size=batch_size,  num_workers=num_workers) # create your dataloader




device = torch.device(config["device"])

model, preprocess, _ = load_open_clip_model()
model = MLP(1024).to(device)

optimizer = torch.optim.Adam(model.parameters())

# choose the loss you want to optimze for
criterion = nn.MSELoss()
criterion2 = nn.L1Loss()

epochs = config["epoch"]

model.train()
best_loss =999
output_name = config["output_scorer"]
save_name = f"{output_name}.pth"


for epoch in range(epochs):
    losses = []
    losses2 = []
    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        losses.append(loss.item())


        optimizer.step()

        if batch_num % 1000 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
            #print(y)

    print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))
    losses = []
    losses2 = []

    for batch_num, input_data in enumerate(val_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        lossMAE = criterion2(output, y)
        #loss.backward()
        losses.append(loss.item())
        losses2.append(lossMAE.item())
        #optimizer.step()

        if batch_num % 1000 == 0:
            print('\tValidation - Epoch %d | Batch %d | MSE Loss %6.2f' % (epoch, batch_num, loss.item()))
            print('\tValidation - Epoch %d | Batch %d | MAE Loss %6.2f' % (epoch, batch_num, lossMAE.item()))

            #print(y)

    print('Validation - Epoch %d | MSE Loss %6.2f' % (epoch, sum(losses)/len(losses)))
    print('Validation - Epoch %d | MAE Loss %6.2f' % (epoch, sum(losses2)/len(losses2)))
    if sum(losses)/len(losses) < best_loss:
        print("Best MAE Val loss so far. Saving model")
        best_loss = sum(losses)/len(losses)
        print( best_loss )

        torch.save(model.state_dict(), save_name )


torch.save(model.state_dict(), save_name)

print( f"best loss: {best_loss} \t savename: {save_name}" )

print("training done")
# inferece test with dummy samples from the val set, sanity check
print( "inferece test with dummy samples from the val set, sanity check")
model.eval()
output = model(x[:5].to(device))
print(output.size())
print(output)
