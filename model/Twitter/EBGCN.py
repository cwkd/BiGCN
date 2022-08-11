import os, sys
sys.path.append(os.getcwd())
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy
from torch.nn import BatchNorm1d
from collections import OrderedDict
sys.path.append(os.getcwd())
from Process.process import *
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from Process.pheme9fold import *
from Process.rand5fold import *
from tools.evaluate import *
from tqdm import tqdm
import argparse


class TDrumorGCN(th.nn.Module):
    def __init__(self, args):
        super(TDrumorGCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(args.input_features, args.hidden_features)
        self.conv2 = GCNConv(args.input_features + args.hidden_features, args.output_features)
        self.device = args.device

        self.num_features_list = [args.hidden_features * r for r in [1]]

        def create_network(self, name):
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                layer_list[name + 'conv{}'.format(l)] = th.nn.Conv1d(
                    in_channels=args.hidden_features,
                    out_channels=args.hidden_features,
                    kernel_size=1,
                    bias=False)
                layer_list[name + 'norm{}'.format(l)] = th.nn.BatchNorm1d(num_features=args.hidden_features)
                layer_list[name + 'relu{}'.format(l)] = th.nn.LeakyReLU()
            layer_list[name + 'conv_out'] = th.nn.Conv1d(in_channels=args.hidden_features,
                                                         out_channels=1,
                                                         kernel_size=1)
            return layer_list

        self.sim_network = th.nn.Sequential(create_network(self, 'sim_val'))
        mod_self = self
        mod_self.num_features_list = [args.hidden_features]
        self.W_mean = th.nn.Sequential(create_network(mod_self, 'W_mean'))
        self.W_bias = th.nn.Sequential(create_network(mod_self, 'W_bias'))
        self.B_mean = th.nn.Sequential(create_network(mod_self, 'B_mean'))
        self.B_bias = th.nn.Sequential(create_network(mod_self, 'B_bias'))
        self.fc1 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.fc2 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.dropout = th.nn.Dropout(args.dropout)
        self.eval_loss = th.nn.KLDivLoss(reduction='batchmean')
        self.bn1 = BatchNorm1d(args.hidden_features + args.input_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        if self.args.edge_infer_td:
            edge_loss, edge_pred = self.edge_infer(x, edge_index)
        else:
            edge_loss, edge_pred = None, None

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        if x.shape[0] != 1:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight=edge_pred)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x, edge_loss

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        x_i = x[row - 1].unsqueeze(2)
        x_j = x[col - 1].unsqueeze(1)
        x_ij = th.abs(x_i - x_j)
        sim_val = self.sim_network(x_ij)
        edge_pred = self.fc1(sim_val)
        edge_pred = th.sigmoid(edge_pred)
        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        logit_mean = w_mean * sim_val + b_mean
        # logit_var = th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias))
        logit_var = th.relu(th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias)))
        edge_y = th.normal(logit_mean, logit_var)
        edge_y = th.sigmoid(edge_y)
        edge_y = self.fc2(edge_y)
        logp_x = F.log_softmax(edge_pred, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)
        edge_loss = self.eval_loss(logp_x, p_y)
        return edge_loss, th.mean(edge_pred, dim=-1).squeeze(1)


class BUrumorGCN(th.nn.Module):
    def __init__(self,args):
        super(BUrumorGCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(args.input_features, args.hidden_features)
        self.conv2 = GCNConv(args.input_features + args.hidden_features, args.output_features)
        self.device = args.device
        self.num_features_list = [args.hidden_features * r for r in [1]]

        def create_network(self, name):
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                layer_list[name + 'conv{}'.format(l)] = th.nn.Conv1d(
                    in_channels=args.hidden_features,
                    out_channels=args.hidden_features,
                    kernel_size=1,
                    bias=False)
                layer_list[name + 'norm{}'.format(l)] = th.nn.BatchNorm1d(num_features=args.hidden_features)
                layer_list[name + 'relu{}'.format(l)] = th.nn.LeakyReLU()
            layer_list[name + 'conv_out'] = th.nn.Conv1d(in_channels=args.hidden_features,
                                                         out_channels=1,
                                                         kernel_size=1)
            return layer_list
        self.sim_network = th.nn.Sequential(create_network(self, 'sim_val'))
        mod_self = self
        mod_self.num_features_list = [args.hidden_features]  #
        self.W_mean = th.nn.Sequential(create_network(mod_self, 'W_mean'))
        self.W_bias = th.nn.Sequential(create_network(mod_self, 'W_bias'))
        self.B_mean = th.nn.Sequential(create_network(mod_self, 'B_mean'))
        self.B_bias = th.nn.Sequential(create_network(mod_self, 'B_bias'))
        self.fc1 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.fc2 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.dropout = th.nn.Dropout(args.dropout)
        self.eval_loss = th.nn.KLDivLoss(reduction='batchmean')  # mean
        self.bn1 = BatchNorm1d(args.hidden_features + args.input_features)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        if self.args.edge_infer_bu:
            edge_loss, edge_pred = self.edge_infer(x, edge_index)
        else:
            edge_loss, edge_pred = None, None

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        if x.shape[0] != 1:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight=edge_pred)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x, edge_loss

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        x_i = x[row - 1].unsqueeze(2)
        x_j = x[col - 1].unsqueeze(1)
        x_ij = th.abs(x_i - x_j)
        sim_val = self.sim_network(x_ij)
        edge_pred = self.fc1(sim_val)
        edge_pred = th.sigmoid(edge_pred)

        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        logit_mean = w_mean * sim_val + b_mean
        # logit_var = th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias))
        logit_var = th.relu(th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias)))
        edge_y = th.normal(logit_mean, logit_var)
        edge_y = th.sigmoid(edge_y)
        edge_y = self.fc2(edge_y)

        logp_x = F.log_softmax(edge_pred, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)
        edge_loss = self.eval_loss(logp_x, p_y)
        return edge_loss, th.mean(edge_pred, dim=-1).squeeze(1)


class EBGCN(th.nn.Module):
    def __init__(self, args):
        super(EBGCN, self).__init__()
        self.args = args
        self.TDrumorGCN = TDrumorGCN(args)
        self.BUrumorGCN = BUrumorGCN(args)
        self.fc = th.nn.Linear((args.hidden_features + args.output_features)*2, args.num_class)

    def forward(self, data):
        TD_x, TD_edge_loss = self.TDrumorGCN(data)
        BU_x, BU_edge_loss = self.BUrumorGCN(data)

        self.x = th.cat((BU_x,TD_x), 1)
        out = self.fc(self.x)
        out = F.log_softmax(out, dim=1)
        return out, TD_edge_loss, BU_edge_loss


def train_GCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs,
              batchsize, datasetname, iter, fold, device, args, **kwargs):

    version = kwargs.get('version', 2)
    log_file_path = kwargs['log_file_path']
    args.input_features = 768
    if datasetname == "PHEME":
        # model = EBGCN(256 * 768, 64, 64).to(device)
        model = EBGCN(args).to(device)
    else:
        # model = EBGCN(5000, 64, 64).to(device)
        model = EBGCN(args).to(device)

    # for module in model.TDrumorGCN.modules():
    #     print(module)

    TD_params = list(map(id, model.TDrumorGCN.conv1.parameters()))
    TD_params += list(map(id, model.TDrumorGCN.conv2.parameters()))
    BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params = filter(lambda p: id(p) not in BU_params + TD_params, model.parameters())
    optimizer = th.optim.Adam([
        {'params': base_params},
        {'params': model.BUrumorGCN.conv1.parameters(), 'lr': args.lr / args.lr_scale_bu},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': args.lr / args.lr_scale_bu},
        {'params': model.TDrumorGCN.conv1.parameters(), 'lr': args.lr / args.lr_scale_td},
        {'params': model.TDrumorGCN.conv2.parameters(), 'lr': args.lr / args.lr_scale_td}
    ], lr=args.lr, weight_decay=args.l2)

    # BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
    # BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    # base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
    # optimizer = th.optim.Adam([
    #     {'params': base_params},
    #     {'params': model.BUrumorGCN.conv1.parameters(), 'lr': lr/5},
    #     {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    # ], lr=lr, weight_decay=weight_decay)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    try:
        for epoch in range(n_epochs):
            model.train()
            traindata_list, testdata_list = loadBiData(datasetname,
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
                if version == 2:  # Version 2
                    new_x = Batch_data.x
                    new_x = new_x.reshape(new_x.shape[0], -1, 768)
                    new_x = new_x[:, 0]
                    Batch_data.x = new_x
                # print(model.TDrumorGCN.conv1.lin(Batch_data.x).shape)
                out_labels, TD_edge_loss, BU_edge_loss = model(Batch_data)
                finalloss = F.nll_loss(out_labels, Batch_data.y)
                loss = finalloss
                if TD_edge_loss is not None:
                    loss += args.edge_loss_td * TD_edge_loss
                if BU_edge_loss is not None:
                    loss += args.edge_loss_bu * BU_edge_loss
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
                if version == 2:  # Version 2
                    new_x = Batch_data.x
                    new_x = new_x.reshape(new_x.shape[0], -1, 768)
                    new_x = new_x[:, 0]
                    Batch_data.x = new_x
                val_out, _, _ = model(Batch_data)
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
            early_stopping(np.mean(temp_val_losses), accs, F1, F2, F3, F4, model, f'EBGCNv{version}', datasetname,
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
                # save_path = os.path.join(save_dir, f'ebgcn_f{fold}_i{iter}_e{epoch:05d}_l{loss:.5f}.pt')
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
            # save_path = os.path.join(save_dir, f'ebgcn_f{fold}_i{iter}_e{epoch:05d}_l{loss:.5f}.pt')
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
            save_path = os.path.join(save_dir, f'final_ebgcn_f{fold}_i{iter}_e{epoch:05d}_l{loss:.5f}.pt')
            th.save(checkpoint, save_path)
    except KeyboardInterrupt:
        # Added model snapshot saving
        checkpoint = {
            'iter': iter,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        root_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(root_dir, 'checkpoints', datasetname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'interrupt_ebgcn_f{fold}_i{iter}_e{epoch:05d}_last.pt')
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetname', type=str, default="Twitter16", metavar='dataname',
                        help='dataset name')
    parser.add_argument('--modelname', type=str, default="BiGCN", metavar='modeltype',
                        help='model type, option: BiGCN/EBGCN')
    parser.add_argument('--input_features', type=int, default=5000, metavar='inputF',
                        help='dimension of input features (TF-IDF)')
    parser.add_argument('--hidden_features', type=int, default=64, metavar='graph_hidden',
                        help='dimension of graph hidden state')
    parser.add_argument('--output_features', type=int, default=64, metavar='output_features',
                        help='dimension of output features')
    parser.add_argument('--num_class', type=int, default=4, metavar='numclass',
                        help='number of classes')
    parser.add_argument('--num_workers', type=int, default=0, metavar='num_workers',
                        help='number of workers for training')

    # Parameters for training the model
    parser.add_argument('--seed', type=int, default=2020, help='random state seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='does not use GPU')
    parser.add_argument('--num_cuda', type=int, default=0,
                        help='index of GPU 0/1')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_scale_bu', type=int, default=5, metavar='LRSB',
                        help='learning rate scale for bottom-up direction')
    parser.add_argument('--lr_scale_td', type=int, default=1, metavar='LRST',
                        help='learning rate scale for top-down direction')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='L2',
                        help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--patience', type=int, default=10, metavar='patience',
                        help='patience for early stop')
    parser.add_argument('--batchsize', type=int, default=128, metavar='BS',
                        help='batch size')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='E',
                        help='number of max epochs')
    parser.add_argument('--iterations', type=int, default=50, metavar='F',
                        help='number of iterations for 5-fold cross-validation')

    # Parameters for the proposed model
    parser.add_argument('--TDdroprate', type=float, default=0.2, metavar='TDdroprate',
                        help='drop rate for edges in the top-down propagation graph')
    parser.add_argument('--BUdroprate', type=float, default=0.2, metavar='BUdroprate',
                        help='drop rate for edges in the bottom-up dispersion graph')
    parser.add_argument('--edge_infer_td', action='store_true', default=True,  # default=False,
                        help='edge inference in the top-down graph')
    parser.add_argument('--edge_infer_bu', action='store_true', default=True,
                        help='edge inference in the bottom-up graph')
    parser.add_argument('--edge_loss_td', type=float, default=0.2, metavar='edge_loss_td',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the top-down propagation graph')
    parser.add_argument('--edge_loss_bu', type=float, default=0.2, metavar='edge_loss_bu',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the bottom-up dispersion graph')
    parser.add_argument('--edge_num', type=int, default=2, metavar='edgenum',
                        help='latent relation types T in the edge inference')

    args = parser.parse_args()
    # if "pheme" in args.datasetname.lower():
    #     args.input_features = 768 * 256  # bert shape.
    #     args.batchsize = 12  # explosion prevention
    #     args.iterations = 1  # just not do that iterations thing.

    lr = 0.0005
    weight_decay = 1e-4
    patience = 10
    n_epochs = 200
    batchsize = 128
    TDdroprate = 0.2
    BUdroprate = 0.2
    # datasetname=sys.argv[1] #"Twitter15"、"Twitter16", 'PHEME'
    datasetname = 'PHEME'
    # iterations=int(sys.argv[2])
    iterations = 1
    if datasetname == 'PHEME':
        batchsize = 24  # 24
    model = "GCN"
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    args.datasetname = 'PHEME'
    args.input_features = 256 * 768
    args.batchsize = 24  # 24
    args.iterations = 1
    # print(args.edge_infer_td, args.edge_infer_bu)
    args.device = device
    log_file_path = f'ebgcn_log.txt'
    for iter in range(iterations):
        if datasetname != 'PHEME':
            fold0_x_test, fold0_x_train, \
            fold1_x_test,  fold1_x_train,  \
            fold2_x_test, fold2_x_train, \
            fold3_x_test, fold3_x_train, \
            fold4_x_test, fold4_x_train = load5foldData(datasetname)
            treeDic=loadTree(datasetname)
            train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
                                                                                                       fold0_x_test,
                                                                                                       fold0_x_train,
                                                                                                       TDdroprate,BUdroprate,
                                                                                                       lr, weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter,
                                                                                                       fold=0,
                                                                                                       device=device)
            train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic,
                                                                                                       fold1_x_test,
                                                                                                       fold1_x_train,
                                                                                                       TDdroprate,BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter,
                                                                                                       fold=1,
                                                                                                       device=device)
            train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic,
                                                                                                       fold2_x_test,
                                                                                                       fold2_x_train,
                                                                                                       TDdroprate,BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter,
                                                                                                       fold=2,
                                                                                                       device=device)
            train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic,
                                                                                                       fold3_x_test,
                                                                                                       fold3_x_train,
                                                                                                       TDdroprate,BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter,
                                                                                                       fold=3,
                                                                                                       device=device)
            train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic,
                                                                                                       fold4_x_test,
                                                                                                       fold4_x_train,
                                                                                                       TDdroprate,BUdroprate, lr,
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
                                   iter,
                                   fold=fold_num,
                                   device=device,
                                   args=args, log_file_path=log_file_path)
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