import sys, os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
# from torch.utils.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from Process.pheme9fold import *
from tools.evaluate import *
import copy


class LSTM(th.nn.Module):
    def __init__(self, in_feats=768, hid_feats=768, out_feats=4, num_layers=2,
                 device=None, pooling='max', version=0):
        super(LSTM, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_feats, hid_feats,
                            num_layers=num_layers, batch_first=True, bidirectional=True, bias=False)
        self.fc = nn.Linear(hid_feats * num_layers * 2, out_feats)
        self.device = device
        self.version = version

    def forward(self, data):
        if self.version == 2:
            new_x = data
        else:
            if self.pooling == 'max':
                new_x = th.nn.MaxPool1d(256)(data.transpose(2, 1)).squeeze(-1).unsqueeze(0)
            elif self.pooling == 'mean':
                new_x = th.nn.AvgPool1d(256)(data.transpose(2, 1)).squeeze(-1).unsqueeze(0)
        # new_x = data[:, 0, :].squeeze(1).unsqueeze(0)
        assert new_x.shape[-1] == self.in_feats

        output, (h_n, c_n) = self.lstm(new_x)
        # print(self.fc)
        # print(output.shape, h_n.shape, h_n.reshape(-1, self.hid_feats).shape)
        logits = self.fc(h_n.reshape(-1, self.hid_feats * self.num_layers * 2))
        logits = F.log_softmax(logits, dim=1)
        return logits


def collate_fn(data):
    print(data)


def train_LSTM(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs,
               batchsize, dataname, iter, fold, device, **kwargs):

    version = kwargs.get('version', 2)
    # pooling = kwargs.get('pooling', 'max')
    log_file_path = kwargs['log_file_path']
    if datasetname == "PHEME":
        model = LSTM(768, 768, 4, 2, device).to(device)
    else:
        model = LSTM(5000, 64, 64, 2, device).to(device)

    # BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
    # BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    # base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
    # optimizer = th.optim.Adam([
    #     {'params': base_params},
    #     {'params': model.BUrumorGCN.conv1.parameters(), 'lr': lr/5},
    #     {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    # ], lr=lr, weight_decay=weight_decay)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    try:
        for epoch in range(n_epochs):
            model.train()
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
            for Batch_data, tweetid in tqdm_train_loader:
                # print(Batch_data, tweetid)
                Batch_data.to(device)
                x = Batch_data.x
                rootindices = Batch_data.rootindex.tolist()
                targets = Batch_data.y
                out_labels = th.zeros(targets.shape[0], 4).to(device)
                for slice_num, rootindex in enumerate(rootindices):
                    if slice_num == len(rootindices) - 1:
                        if version == 2:
                            slice = Batch_data.cls[rootindex:]
                        else:
                            slice = x[rootindex:]
                    else:
                        if version == 2:
                            slice = Batch_data.cls[rootindex:rootindices[slice_num + 1]]
                        else:
                            slice = x[rootindex:rootindices[slice_num + 1]]
                    if version != 2:
                        slice = slice.reshape(slice.shape[0], -1, 768)
                    # print(slice.shape)
                    slice_out_labels = model(slice)
                    # print(out_labels)
                    out_labels[slice_num] = slice_out_labels
                kwargs['debug'][0] = out_labels
                finalloss = F.nll_loss(out_labels, Batch_data.y)
                loss = finalloss
                optimizer.zero_grad()
                loss.backward()
                avg_loss.append(loss.item())
                optimizer.step()
                _, pred = out_labels.max(dim=-1)
                correct = pred.eq(Batch_data.y).sum().item()
                train_acc = correct / len(Batch_data.y)
                avg_acc.append(train_acc)
                print("Fold {} | Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(
                    fold, iter, epoch, batch_idx, loss.item(), train_acc))
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
            for Batch_data, tweetid in tqdm_test_loader:
                Batch_data.to(device)
                x = Batch_data.x
                rootindices = Batch_data.rootindex.tolist()
                targets = Batch_data.y
                val_out = th.zeros(targets.shape[0], 4).to(device)
                for slice_num, rootindex in enumerate(rootindices):
                    if slice_num == len(rootindices) - 1:
                        if version == 2:
                            slice = Batch_data.cls[rootindex:]
                        else:
                            slice = x[rootindex:]
                    else:
                        if version == 2:
                            slice = Batch_data.cls[rootindex:rootindices[slice_num + 1]]
                        else:
                            slice = x[rootindex:rootindices[slice_num + 1]]
                    if version != 2:
                        slice = slice.reshape(slice.shape[0], -1, 768)
                    slice_val_out = model(slice)
                    # print(out_labels)
                    val_out[slice_num] = slice_val_out
                # val_out = model(Batch_data)
                val_loss = F.nll_loss(val_out, Batch_data.y)
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
            print("Fold {} | Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(fold, epoch, np.mean(temp_val_losses),
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
            with open(log_file_path, 'a') as f:
                f.write(f'Fold: {fold}| Iter: {iter:03d} | Epoch {epoch:05d} | '
                        f'Val_loss {np.mean(temp_val_losses):.4f} | Val_acc: {np.mean(temp_val_accs):.4f}'
                        f'Results: {res}\n')
            checkpoint = {
                'fold': fold,
                'iter': iter,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_losses[-1],
                'res': res
            }
            accs = np.mean(temp_val_accs)
            F1 = np.mean(temp_val_F1)
            F2 = np.mean(temp_val_F2)
            F3 = np.mean(temp_val_F3)
            F4 = np.mean(temp_val_F4)
            early_stopping(np.mean(temp_val_losses), accs, F1, F2, F3, F4, model, f'LSTM', datasetname,
                           checkpoint=checkpoint)

            if early_stopping.early_stop:
                print("Early stopping")
                accs = early_stopping.accs
                F1 = early_stopping.F1
                F2 = early_stopping.F2
                F3 = early_stopping.F3
                F4 = early_stopping.F4
                # Added model snapshot saving
                # checkpoint = {
                #     'iter': iter,
                #     'epoch': epoch,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': train_losses[-1],
                #     'res': res
                # }
                # root_dir = os.path.dirname(os.path.abspath(__file__))
                # save_dir = os.path.join(root_dir, 'checkpoints', datasetname)
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                # save_path = os.path.join(save_dir, f'{pooling}treebert_f{fold}_i{iter}_e{epoch:05d}_l{loss:.5f}.pt')
                # th.save(checkpoint, save_path)
                return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4
            # checkpoint = {
            #     'fold': fold,
            #     'iter': iter,
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': train_losses[-1],
            #     'res': res
            # }
            # root_dir = os.path.dirname(os.path.abspath(__file__))
            # save_dir = os.path.join(root_dir, 'checkpoints', datasetname)
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # save_path = os.path.join(save_dir, f'{pooling}treebert_f{fold}_i{iter}_e{epoch:05d}_l{loss:.5f}.pt')
            # th.save(checkpoint, save_path)
        else:
            # checkpoint = {
            #     'fold': fold,
            #     'iter': iter,
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': train_losses[-1],
            #     'res': res
            # }
            root_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(root_dir, 'checkpoints', datasetname)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f'final_LSTM_f{fold}_i{iter}_e{epoch:05d}_l{loss:.5f}.pt')
            th.save(checkpoint, save_path)
    except KeyboardInterrupt:
        # Added model snapshot saving
        checkpoint = {
            'fold': fold,
            'iter': iter,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        root_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(root_dir, 'checkpoints', datasetname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'interrupt_LSTM_f{fold}_i{iter}_e{epoch:05d}_last.pt')
        th.save(checkpoint, save_path)
    # except:
    #     t = th.cuda.get_device_properties(0).total_memory
    #     r = th.cuda.memory_reserved(0)
    #     a = th.cuda.memory_allocated(0)
    #     f = r - a  # free inside reserved
    #     print(f'{e}\n')
    #     print(Batch_data)
    #     print(f'GPU Memory:\nTotal: {t}\tReserved: {r}\tAllocated: {a}\tFree: {f}\n')
    return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4


if __name__ == '__main__':
    lr = 0.0005
    weight_decay = 1e-4
    patience = 10
    n_epochs = 200
    batchsize = 128
    TDdroprate = 0  # 0.2
    BUdroprate = 0  # 0.2
    # datasetname=sys.argv[1] #"Twitter15"、"Twitter16", 'PHEME'
    datasetname ='PHEME'
    # iterations=int(sys.argv[2])
    if datasetname == 'PHEME':
        batchsize = 24  # 24
    iterations = 1
    model = "LSTM"
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    debug = [None]
    # pooling = 'max'
    log_file_path = f'{model}_log.txt'
    for iter in range(iterations):
        if datasetname != 'PHEME':
            fold0_x_test, fold0_x_train, \
            fold1_x_test, fold1_x_train, \
            fold2_x_test, fold2_x_train, \
            fold3_x_test, fold3_x_train, \
            fold4_x_test, fold4_x_train = load5foldData(datasetname)
            treeDic=loadTree(datasetname)
            train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_LSTM(treeDic,
                                                                                                        fold0_x_test,
                                                                                                        fold0_x_train,
                                                                                                        TDdroprate, BUdroprate,
                                                                                                        lr, weight_decay,
                                                                                                        patience,
                                                                                                        n_epochs,
                                                                                                        batchsize,
                                                                                                        datasetname,
                                                                                                        iter,
                                                                                                        fold=0,
                                                                                                        device=device)
            train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_LSTM(treeDic,
                                                                                                        fold1_x_test,
                                                                                                        fold1_x_train,
                                                                                                        TDdroprate, BUdroprate, lr,
                                                                                                        weight_decay,
                                                                                                        patience,
                                                                                                        n_epochs,
                                                                                                        batchsize,
                                                                                                        datasetname,
                                                                                                        iter,
                                                                                                        fold=1,
                                                                                                        device=device)
            train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_LSTM(treeDic,
                                                                                                        fold2_x_test,
                                                                                                        fold2_x_train,
                                                                                                        TDdroprate, BUdroprate, lr,
                                                                                                        weight_decay,
                                                                                                        patience,
                                                                                                        n_epochs,
                                                                                                        batchsize,
                                                                                                        datasetname,
                                                                                                        iter,
                                                                                                        fold=2,
                                                                                                        device=device)
            train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_LSTM(treeDic,
                                                                                                        fold3_x_test,
                                                                                                        fold3_x_train,
                                                                                                        TDdroprate, BUdroprate, lr,
                                                                                                        weight_decay,
                                                                                                        patience,
                                                                                                        n_epochs,
                                                                                                        batchsize,
                                                                                                        datasetname,
                                                                                                        iter,
                                                                                                        fold=3,
                                                                                                        device=device)
            train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_LSTM(treeDic,
                                                                                                        fold4_x_test,
                                                                                                        fold4_x_train,
                                                                                                        TDdroprate, BUdroprate, lr,
                                                                                                        weight_decay,
                                                                                                        patience,
                                                                                                        n_epochs,
                                                                                                        batchsize,
                                                                                                        datasetname,
                                                                                                        iter,
                                                                                                        fold=4,
                                                                                                        device=device)
            test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
            NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
            FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
            TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
            UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
            print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
                sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))
        else:
            treeDic = None
            train_losses_dict, val_losses_dict, train_accs_dict, val_accs_dict = {}, {}, {}, {}
            test_accs_dict, F1_dict, F2_dict, F3_dict, F4_dict = {}, {}, {}, {}, {}
            for fold_num, (fold_train, fold_test) in enumerate(load9foldData(datasetname)):
                output = train_LSTM(treeDic,
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
                                    iter,
                                    fold=fold_num,
                                    device=device, debug=debug, log_file_path=log_file_path)
                train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4 = output
                train_losses_dict[f'train_losses_{fold_num}'] = train_losses
                val_losses_dict[f'val_losses_{fold_num}'] = val_losses
                train_accs_dict[f'train_accs_{fold_num}'] = train_accs
                val_accs_dict[f'val_accs_{fold_num}'] = val_accs
                test_accs_dict[f'test_accs_{fold_num}'] = accs
                F1_dict[f'F1_{fold_num}'] = F1
                F2_dict[f'F2_{fold_num}'] = F2
                F3_dict[f'F3_{fold_num}'] = F3
                F4_dict[f'F4_{fold_num}'] = F4
            test_accs.append(sum([v for k, v in test_accs_dict.items()]) / 5)
            NR_F1.append(sum([v for k, v in F1_dict.items()]) / 5)
            FR_F1.append(sum([v for k, v in F2_dict.items()]) / 5)
            TR_F1.append(sum([v for k, v in F3_dict.items()]) / 5)
            UR_F1.append(sum([v for k, v in F4_dict.items()]) / 5)
            summary = "Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
                sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations,
                sum(UR_F1) / iterations)
            print(summary)
            with open(log_file_path, 'a') as f:
                f.write(f'{summary}\n')


