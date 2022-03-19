import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy
from model.Twitter.BiGCN_Twitter import Net


def train_GCN(treeDic, x_test, x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,fold):
    traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)
    print(traindata_list, testdata_list)
    for i, data in enumerate(traindata_list):
        print(i, data)
        if i == 100:
            break


if __name__ == "__main__":
    lr = 0.0005
    weight_decay = 1e-4
    patience = 10
    n_epochs = 200
    batchsize = 128
    TDdroprate = 0.2
    BUdroprate = 0.2
    datasetname="Twitter15"
    fold0_x_test, fold0_x_train, \
    fold1_x_test, fold1_x_train, \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test, fold4_x_train = load5foldData(datasetname)
    treeDic = loadTree(datasetname)
    train_GCN(treeDic,
              fold0_x_test,
              fold0_x_train,
              TDdroprate,
              BUdroprate,
              lr,
              weight_decay,
              patience,
              n_epochs,
              batchsize,
              datasetname,
              iter=0,
              fold=0)