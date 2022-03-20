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
from Process.pheme9fold import *


def train_GCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs,
              batchsize, dataname, iter, fold):
    traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate, BUdroprate)
    print(traindata_list, testdata_list)
    for i, data in enumerate(traindata_list):
        print(i, data)
        if i == 100:
            break
    if datasetname == "PHEME":
        model = Net(256*768,64,64).to(device)
    else:
        model = Net(5000,64,64).to(device)

    BU_params=list(map(id,model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    optimizer = th.optim.Adam([
        {'params': base_params},
        {'params': model.BUrumorGCN.conv1.parameters(), 'lr': lr/5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    try:
        for epoch in range(n_epochs):
            traindata_list, testdata_list = loadBiData(dataname,
                                                       treeDic,
                                                       x_train,
                                                       x_test,
                                                       TDdroprate,
                                                       BUdroprate)
            train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
            test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
            avg_loss = []
            avg_acc = []
            batch_idx = 0
            tqdm_train_loader = tqdm(train_loader)
            for Batch_data in tqdm_train_loader:
                Batch_data.to(device)
                out_labels= model(Batch_data)
                finalloss=F.nll_loss(out_labels,Batch_data.y)
                loss=finalloss
                optimizer.zero_grad()
                loss.backward()
                avg_loss.append(loss.item())
                optimizer.step()
                _, pred = out_labels.max(dim=-1)
                correct = pred.eq(Batch_data.y).sum().item()
                train_acc = correct / len(Batch_data.y)
                avg_acc.append(train_acc)
                print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                     loss.item(),
                                                                                                     train_acc))
                batch_idx = batch_idx + 1

            train_losses.append(np.mean(avg_loss))
            train_accs.append(np.mean(avg_acc))

            temp_val_losses = []
            temp_val_accs = []
            temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
            temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
            temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
            temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            model.eval()
            tqdm_test_loader = tqdm(test_loader)
            for Batch_data in tqdm_test_loader:
                Batch_data.to(device)
                val_out = model(Batch_data)
                val_loss  = F.nll_loss(val_out, Batch_data.y)
                temp_val_losses.append(val_loss.item())
                _, val_pred = val_out.max(dim=1)
                correct = val_pred.eq(Batch_data.y).sum().item()
                val_acc = correct / len(Batch_data.y)
                Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                    val_pred, Batch_data.y)
                temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                    Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
                temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                    Recll2), temp_val_F2.append(F2), \
                temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                    Recll3), temp_val_F3.append(F3), \
                temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                    Recll4), temp_val_F4.append(F4)
                temp_val_accs.append(val_acc)
            val_losses.append(np.mean(temp_val_losses))
            val_accs.append(np.mean(temp_val_accs))
            print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                               np.mean(temp_val_accs)))

            res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
                   'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                           np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
                   'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                           np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
                   'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                           np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
                   'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                           np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
            print('results:', res)

            early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                           np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', dataname)
            accs =np.mean(temp_val_accs)
            F1 = np.mean(temp_val_F1)
            F2 = np.mean(temp_val_F2)
            F3 = np.mean(temp_val_F3)
            F4 = np.mean(temp_val_F4)
            if early_stopping.early_stop:
                print("Early stopping")
                accs=early_stopping.accs
                F1=early_stopping.F1
                F2 = early_stopping.F2
                F3 = early_stopping.F3
                F4 = early_stopping.F4
                raise Exception
    except:
        # Added model snapshot saving
        checkpoint = {
            'iter': iter,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'res': res
        }
        root_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(root_dir, 'checkpoints')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'bigcn_f{fold}_i{iter}_e{epoch:05d}_l{loss:.5f}.pt')
        th.save(checkpoint, save_path)
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4


if __name__ == "__main__":
    lr = 0.0005
    weight_decay = 1e-4
    patience = 10
    n_epochs = 200
    batchsize = 24
    TDdroprate = 0.2
    BUdroprate = 0.2
    datasetname="PHEME"
    iterations = 10
    model = "GCN"
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    # fold0_x_test, fold0_x_train, \
    # fold1_x_test, fold1_x_train, \
    # fold2_x_test, fold2_x_train, \
    # fold3_x_test, fold3_x_train, \
    # fold4_x_test, fold4_x_train = load5foldData('Twitter15')
    # treeDic = loadTree(datasetname)
    treeDic = None

    for fold_num, (fold_train, fold_test) in enumerate(load9foldData(datasetname)):
        output = train_GCN(treeDic,
                           fold_test,
                           fold_train,
                           TDdroprate,
                           BUdroprate,
                           lr,
                           weight_decay,
                           patience,
                           n_epochs,
                           batchsize,
                           datasetname,
                           iter=0,
                           fold=fold_num)
        train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = output