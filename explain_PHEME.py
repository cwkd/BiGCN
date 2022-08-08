import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import torch_sparse
# from torch_scatter.utils import broadcast
from torch_scatter import scatter_mean
# import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.nn.conv import gcn_conv

from model.Twitter.BiGCN_Twitter import Net
from model.Twitter.BERT_Twitter import TreeBERT
from model.Twitter.EBGCN import EBGCN
from Process.process import loadBiData, loadTree
from Process.pheme9fold import load9foldData

# from torch_geometric.utils import add_remaining_self_loops
# from torch_geometric.utils.num_nodes import maybe_num_nodes

import lrp_pytorch.modules.utils as lrp_utils
from tqdm import tqdm

import argparse
import json

FOLD_2_EVENTNAME = {0: 'charliehebdo',
                    1: 'ebola',
                    2: 'ferguson',
                    3: 'germanwings',
                    4: 'gurlitt',
                    5: 'ottawashooting',
                    6: 'prince',
                    7: 'putinmissing',
                    8: 'sydneysiege'}
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXPLAIN_DIR = os.path.join(DATA_DIR, 'explain')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'model', 'Twitter', 'checkpoints')
random.seed(0)

LRP_PARAMS = {
    'linear_eps': 1e-6
}


def swap_elements(lst, idx1, idx2):
    if isinstance(idx1, int):
        lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
        return lst
    for i, j in zip(idx1, idx2):
        lst[i], lst[j] = lst[j], lst[i]
    return lst


def extract_gcn_conv(conv: gcn_conv.GCNConv, x, edge_index, edge_weight=None):

    edge_index, edge_weight = gcn_conv.gcn_norm(  # yapf: disable
        edge_index, edge_weight, x.size(-2), False, True)

    temp_linear = None
    for name, mod in conv.named_modules():
        # print('\n', name, mod)
        temp = lrp_utils.get_lrpwrappermodule(mod, LRP_PARAMS)
        if temp is not None:
            temp_linear = temp

    # print(temp_linear.__dict__)
    x.requires_grad = True
    with torch.enable_grad():
        temp_output = temp_linear(x)

    print(temp_output.shape)
    # temp_output[0].backward()
    # rel = x.grad.data
    # print(rel)
    # print(rel.shape)
    print('\nedge_index\n', edge_index.shape)
    print('\nedge_weight\n', edge_weight.shape)
    # print('\nedge_weight_topk\n', torch.topk(edge_weight, edge_weight.shape[0]))
    # print(torch.topk(edge_weight, edge_weight.shape[0]))
    raise Exception


def extract_intermediates_bigcn(conv1, conv2, x, edge_index, data_sample, device):
    gcn_explanations = {}
    # Need to fix this
    # extract_gcn_conv(conv1, x, edge_index)
    conv1_output = conv1(x, edge_index)
    conv1_output_sum = torch.sum(conv1_output, dim=-1)
    # print('conv1_output_sum', conv1_output_sum)
    conv1_output_sum_topk = torch.topk(conv1_output_sum,
                                       k=conv1_output_sum.shape[0])
    # print('conv1_output_sum_topk', conv1_output_sum_topk)
    gcn_explanations['conv1_output_sum_topk'] = \
        [conv1_output_sum_topk.indices.tolist(),
         conv1_output_sum_topk.values.tolist()]

    rootindex = data_sample.rootindex
    root_extend = torch.zeros(len(data_sample.batch), x.size(1)).to(device)
    batch_size = max(data_sample.batch) + 1
    for num_batch in range(batch_size):
        index = (torch.eq(data_sample.batch, num_batch))
        root_extend[index] = x[rootindex[num_batch]]
    conv1_output_cat_x = torch.cat((conv1_output, root_extend), 1)
    conv1_output_cat_x_sum = torch.sum(conv1_output_cat_x, dim=-1)
    # print('conv1_output_cat_x_sum', conv1_output_cat_x_sum)
    conv1_output_cat_x_sum_topk = torch.topk(conv1_output_cat_x_sum,
                                             k=conv1_output_cat_x_sum.shape[0])
    # print('conv1_output_cat_x_sum_topk', conv1_output_cat_x_sum_topk)
    gcn_explanations['conv1_output_cat_x_sum_topk'] = \
        [conv1_output_cat_x_sum_topk.indices.tolist(),
         conv1_output_cat_x_sum_topk.values.tolist()]

    conv1_output_cat_x_relu = F.relu(conv1_output_cat_x)
    conv2_input = F.dropout(conv1_output_cat_x_relu, training=False)
    conv2_input_sum = torch.sum(conv2_input, dim=-1)
    # print('conv2_input_sum', conv2_input_sum)
    conv2_input_sum_topk = torch.topk(conv2_input_sum,
                                      k=conv2_input_sum.shape[0])
    # print('conv2_input_sum_topk', conv2_input_sum_topk)
    gcn_explanations['conv2_input_sum_topk'] = \
        [conv2_input_sum_topk.indices.tolist(),
         conv2_input_sum_topk.values.tolist()]

    conv2_output = conv2(conv2_input, edge_index)
    conv2_output_sum = torch.sum(conv2_output, dim=-1)
    # print('conv2_output_sum', conv2_output_sum)
    conv2_output_sum_topk = torch.topk(conv2_output_sum,
                                              k=conv2_output_sum.shape[0])
    # print('conv2_output_sum_topk', conv2_output_sum_topk)
    gcn_explanations['conv2_output_sum_topk'] = \
        [conv2_output_sum_topk.indices.tolist(),
         conv2_output_sum_topk.values.tolist()]

    conv2_output_relu = F.relu(conv2_output)

    root_extend2 = torch.zeros(len(data_sample.batch), conv1_output.size(1)).to(device)
    for num_batch in range(batch_size):
        index = (torch.eq(data_sample.batch, num_batch))
        root_extend2[index] = conv1_output[rootindex[num_batch]]
    scatter_mean_input = torch.cat((conv2_output_relu, root_extend2), 1)
    scatter_mean_input_sum = torch.sum(scatter_mean_input, dim=-1)
    # print('scatter_mean_input_sum', scatter_mean_input_sum)
    scatter_mean_input_sum_topk = torch.topk(scatter_mean_input_sum,
                                                    k=scatter_mean_input_sum.shape[0])
    gcn_explanations['scatter_mean_input_sum_topk'] = \
        [scatter_mean_input_sum_topk.indices.tolist(),
         scatter_mean_input_sum_topk.values.tolist()]

    gcn_output = scatter_mean(scatter_mean_input, data_sample.batch, dim=0)
    # gcn_output_sum = torch.sum(gcn_output, dim=-1)
    # print('gcn_output', gcn_output)
    gcn_explanations['gcn_output'] = gcn_output.tolist()

    return gcn_explanations


def extract_intermediates_ebgcn(gcn, x, edge_index, data_sample, device):
    gcn_explanations = {}

    conv1_output = gcn.conv1(x, edge_index)
    conv1_output_sum = torch.sum(conv1_output, dim=-1)
    conv1_output_sum_topk = torch.topk(conv1_output_sum,
                                       k=conv1_output_sum.shape[0])
    gcn_explanations['conv1_output_sum_topk'] = \
        [conv1_output_sum_topk.indices.tolist(),
         conv1_output_sum_topk.values.tolist()]

    edge_loss, edge_pred = gcn.edge_infer(conv1_output, edge_index)
    # if gcn.args.edge_infer_td:
    #     edge_loss, edge_pred = gcn.edge_infer(x, edge_index)
    # else:
    #     edge_loss, edge_pred = None, None
    # edge_loss, edge_pred = None, None

    rootindex = data_sample.rootindex
    root_extend = torch.zeros(len(data_sample.batch), x.size(1)).to(device)
    batch_size = max(data_sample.batch) + 1
    for num_batch in range(batch_size):
        index = (torch.eq(data_sample.batch, num_batch))
        root_extend[index] = x[rootindex[num_batch]]
    conv1_output_cat_x = torch.cat((conv1_output, root_extend), 1)

    conv1_output_cat_x_sum = torch.sum(conv1_output_cat_x, dim=-1)
    conv1_output_cat_x_sum_topk = torch.topk(conv1_output_cat_x_sum,
                                             k=conv1_output_cat_x_sum.shape[0])
    gcn_explanations['conv1_output_cat_x_sum_topk'] = \
        [conv1_output_cat_x_sum_topk.indices.tolist(),
         conv1_output_cat_x_sum_topk.values.tolist()]

    conv1_output_cat_x_bn1 = gcn.bn1(conv1_output_cat_x)
    conv1_output_cat_x_bn1_sum = torch.sum(conv1_output_cat_x_bn1, dim=-1)
    conv1_output_cat_x_bn1_sum_topk = torch.topk(conv1_output_cat_x_bn1_sum,
                                             k=conv1_output_cat_x_bn1_sum.shape[0])
    gcn_explanations['conv1_output_cat_x_bn1_sum_topk'] = \
        [conv1_output_cat_x_bn1_sum_topk.indices.tolist(),
         conv1_output_cat_x_bn1_sum_topk.values.tolist()]

    conv1_output_cat_x_bn1_relu = F.relu(conv1_output_cat_x_bn1)

    conv2_input = conv1_output_cat_x_bn1_relu
    conv2_input_sum = torch.sum(conv2_input, dim=-1)
    conv2_input_sum_topk = torch.topk(conv2_input_sum,
                                      k=conv2_input_sum.shape[0])
    gcn_explanations['conv2_input_sum_topk'] = \
        [conv2_input_sum_topk.indices.tolist(),
         conv2_input_sum_topk.values.tolist()]

    conv2_output = gcn.conv2(conv2_input, edge_index, edge_weight=edge_pred)
    conv2_output_sum = torch.sum(conv2_output, dim=-1)
    conv2_output_sum_topk = torch.topk(conv2_output_sum,
                                       k=conv2_output_sum.shape[0])
    gcn_explanations['conv2_output_sum_topk'] = \
        [conv2_output_sum_topk.indices.tolist(),
         conv2_output_sum_topk.values.tolist()]

    conv2_output_relu = F.relu(conv2_output)

    root_extend2 = torch.zeros(len(data_sample.batch), conv1_output.size(1)).to(device)
    for num_batch in range(batch_size):
        index = (torch.eq(data_sample.batch, num_batch))
        root_extend2[index] = conv1_output[rootindex[num_batch]]
    conv2_output_relu_cat_conv1_output = torch.cat((conv2_output_relu, root_extend2), 1)
    scatter_mean_input = conv2_output_relu_cat_conv1_output
    scatter_mean_input_sum = torch.sum(scatter_mean_input, dim=-1)
    scatter_mean_input_sum_topk = torch.topk(scatter_mean_input_sum,
                                             k=scatter_mean_input_sum.shape[0])
    gcn_explanations['scatter_mean_input_sum_topk'] = \
        [scatter_mean_input_sum_topk.indices.tolist(),
         scatter_mean_input_sum_topk.values.tolist()]

    gcn_output = scatter_mean(scatter_mean_input, data_sample.batch, dim=0)
    gcn_explanations['gcn_output'] = gcn_output.tolist()

    return gcn_explanations


def enumerate_children(module):
    for child in module.children():
        enumerate_children(child)
        print(child)


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
    parser.add_argument('--TDdroprate', type=float, default=0, metavar='TDdroprate',
                        help='drop rate for edges in the top-down propagation graph')
    parser.add_argument('--BUdroprate', type=float, default=0, metavar='BUdroprate',
                        help='drop rate for edges in the bottom-up dispersion graph')
    parser.add_argument('--edge_infer_td', action='store_true', default=True,  # default=False,
                        help='edge inference in the top-down graph')
    parser.add_argument('--edge_infer_bu', action='store_true', default=True,  # default=True,
                        help='edge inference in the bottom-up graph')
    parser.add_argument('--edge_loss_td', type=float, default=0.2, metavar='edge_loss_td',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the top-down propagation graph')
    parser.add_argument('--edge_loss_bu', type=float, default=0.2, metavar='edge_loss_bu',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the bottom-up dispersion graph')
    parser.add_argument('--edge_num', type=int, default=2, metavar='edgenum',
                        help='latent relation types T in the edge inference')

    args = parser.parse_args()

    # some admin stuff
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TDdroprate = 0  # 0.2
    BUdroprate = 0  # 0.2
    # datasetname = sys.argv[1]  # "Twitter15"、"Twitter16", 'PHEME'
    datasetname='PHEME'
    # iterations = int(sys.argv[2])
    iterations = 1
    if datasetname == 'PHEME':
        batchsize = 1
    args.datasetname = 'PHEME'
    args.input_features = 256 * 768
    args.batchsize = 1
    args.iterations = 1
    args.device = device
    # iterations=10
    model_types = ['BiGCN', 'EBGCN', 'BERT']
    pooling_types = ['max', 'mean']
    randomise_types = [1.0, 0.5, 0.25, 0.0]
    # model = "BiGCN"
    # model = 'EBGCN'
    # model = 'BERT'
    # pooling = 'mean'  # 'max'
    model = model_types[0]
    pooling = pooling_types[0]
    # randomise = randomise_types[2]  # Randomise Nodes
    treeDic = None  # Not required for PHEME

    save_dir_path = 'data/explain/PHEME'
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # init network
    if model == 'BiGCN':
        net = Net(256 * 768, 64, 64, device).to(device)
        checkpoint_paths = ['best_BiGCN_PHEME_f0_i0_e00005_l1.17733.pt',
                            'best_BiGCN_PHEME_f1_i0_e00004_l1.06584.pt',
                            'best_BiGCN_PHEME_f2_i0_e00004_l1.09573.pt',
                            'best_BiGCN_PHEME_f3_i0_e00005_l1.04328.pt',
                            'best_BiGCN_PHEME_f4_i0_e00000_l2.39317.pt',
                            'best_BiGCN_PHEME_f5_i0_e00005_l0.96496.pt',
                            'best_BiGCN_PHEME_f6_i0_e00000_l1.78671.pt',
                            'best_BiGCN_PHEME_f7_i0_e00010_l1.05369.pt',
                            'best_BiGCN_PHEME_f8_i0_e00003_l1.13725.pt']
        # checkpoint_paths = ['bigcn_f0_i5_e00016_l0.35202.pt',
        #                     'bigcn_f1_i0_e00029_l0.45703.pt',
        #                     'bigcn_f2_i0_e00019_l0.78689.pt',
        #                     'bigcn_f3_i4_e00021_l0.76735.pt',
        #                     'bigcn_f4_i5_e00012_l0.31995.pt',
        #                     'bigcn_f5_i7_e00015_l0.19297.pt',
        #                     'bigcn_f6_i0_e00016_l0.01913.pt',
        #                     'bigcn_f7_i0_e00018_l0.15752.pt',
        #                     'bigcn_f8_i6_e00018_l0.69322.pt']
    elif model == 'EBGCN':
        net = EBGCN(args).to(device)
        checkpoint_paths = ['best_EBGCN_PHEME_f0_i0_e00007_l0.62714.pt',
                            'best_EBGCN_PHEME_f1_i0_e00006_l1.77484.pt',
                            'best_EBGCN_PHEME_f2_i0_e00003_l1.00047.pt',
                            'best_EBGCN_PHEME_f3_i0_e00008_l0.94641.pt',
                            'best_EBGCN_PHEME_f4_i0_e00000_l1.81918.pt',
                            'best_EBGCN_PHEME_f5_i0_e00000_l2.73212.pt',
                            'best_EBGCN_PHEME_f6_i0_e00000_l2.18175.pt',
                            'best_EBGCN_PHEME_f7_i0_e00012_l0.35055.pt',
                            'best_EBGCN_PHEME_f8_i0_e00003_l1.76857.pt']
        # checkpoint_paths = ['ebgcnPHEME charliehebdo.m',
        #                     'ebgcnPHEME ebola-essien.m',
        #                     'ebgcnPHEME ferguson.m',
        #                     'ebgcnPHEME germanwings-crash.m',
        #                     'ebgcnPHEME gurlitt.m',
        #                     'ebgcnPHEME ottawashooting.m',
        #                     'ebgcnPHEME prince-toronto.m',
        #                     'ebgcnPHEME putinmissing.m',
        #                     'ebgcnPHEME sydneysiege.m']
        # obj = torch.load(checkpoint_paths[0])
    elif model == 'BERT':
        net = TreeBERT(256 * 768, 64, 64, device, pooling).to(device)
        if pooling == 'max':
            checkpoint_paths = ['best_maxBERT_PHEME_f0_i0_e00007_l1.18175.pt',
                                'best_maxBERT_PHEME_f1_i0_e00004_l1.08462.pt',
                                'best_maxBERT_PHEME_f2_i0_e00004_l1.12532.pt',
                                'best_maxBERT_PHEME_f3_i0_e00007_l1.06524.pt',
                                'best_maxBERT_PHEME_f4_i0_e00011_l1.08704.pt',
                                'best_maxBERT_PHEME_f5_i0_e00007_l1.05438.pt',
                                'best_maxBERT_PHEME_f6_i0_e00015_l1.03478.pt',
                                'best_maxBERT_PHEME_f7_i0_e00009_l1.07684.pt',
                                'best_maxBERT_PHEME_f8_i0_e00002_l1.07145.pt']
            # checkpoint_paths = ['maxtreebert_f0_i0_e00021_l4.40295.pt']
        elif pooling == 'mean':
            checkpoint_paths = ['best_meanBERT_PHEME_f0_i0_e00010_l1.17131.pt',
                                'best_meanBERT_PHEME_f1_i0_e00005_l1.08910.pt',
                                'best_meanBERT_PHEME_f2_i0_e00007_l1.09002.pt',
                                'best_meanBERT_PHEME_f3_i0_e00001_l1.06774.pt',
                                'best_meanBERT_PHEME_f4_i0_e00006_l1.06748.pt',
                                'best_meanBERT_PHEME_f5_i0_e00000_l1.05710.pt',
                                'best_meanBERT_PHEME_f6_i0_e00000_l1.04125.pt',
                                'best_meanBERT_PHEME_f7_i0_e00003_l1.07912.pt',
                                'best_meanBERT_PHEME_f8_i0_e00009_l1.07655.pt']
            # checkpoint_paths = ['meantreebert_f0_i0_e00010_l4.34632.pt']

    # folder_path = os.path.join(EXPLAIN_DIR, datasetname, 'charliehebdo')
    # filenames = list(filter(lambda x: x.find('BERT') != -1, os.listdir(folder_path)))
    # for filename in filenames:
    #     os.remove(os.path.join(folder_path, filename))
    # print(filenames)
    # raise Exception

    # enumerate_children(model)
    # for child in net.children():
    #     # print(child, type(child))
    #     if len(list(child.children())) != 0:
    #         for sub_child in child.children():
    #             # print(sub_child, type(sub_child))
    #             if isinstance(sub_child, torch_geometric.nn.conv.MessagePassing):
    #                 sub_child: torch_geometric.nn.conv.gcn_conv.GCNConv
    #                 # sub_child.explain = True
    outputs = []
    for randomise in randomise_types:
        print(f'\nGenerating:\t'
              f'Model: {model}\t'
              f'Pooling: {pooling if model == "BERT" else None}\t'
              f'Randomise: {randomise}')
        for fold_num, (fold_train, fold_test) in enumerate(load9foldData(datasetname)):
            try:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, datasetname, checkpoint_paths[fold_num])
                checkpoint = torch.load(checkpoint_path)
                # if model == 'EBGCN':
                #     net.load_state_dict(checkpoint)
                # else:
                #     net.load_state_dict(checkpoint['model_state_dict'])
                net.load_state_dict(checkpoint['model_state_dict'])
                print(f'Checkpoint loaded from {checkpoint_path}')
            except:
                print('No checkpoint to load')
            net.eval()
            fold_output = {}
            # print(fold_num)
            event_name = FOLD_2_EVENTNAME[fold_num]
            if not os.path.exists(os.path.join(EXPLAIN_DIR, datasetname, event_name)):
                os.makedirs(os.path.join(EXPLAIN_DIR, datasetname, event_name))
                print(f'Save directory for {event_name} created at: '
                      f'{os.path.join(EXPLAIN_DIR, datasetname, event_name)}\n')
            else:
                print(f'Save directory for {event_name} already exists\n')
            traindata_list, testdata_list = loadBiData(datasetname,
                                                       treeDic,
                                                       fold_train,
                                                       fold_test,
                                                       TDdroprate,
                                                       BUdroprate)
            # train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=False, num_workers=5)
            test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=False, num_workers=5)
            if model != 'BERT':
                evaluation_log_path = os.path.join(EXPLAIN_DIR,
                                                   f'{datasetname}_{event_name}_{model}_r{randomise}_eval.txt')
            else:
                evaluation_log_path = os.path.join(EXPLAIN_DIR,
                                                   f'{datasetname}_{event_name}_{pooling}{model}_r{randomise}_eval.txt')
            eval_log_string = ''
            conf_mat = np.zeros((4, 4))
            for sample_num, (data_sample, root_tweetid) in enumerate(tqdm(test_loader)):
                explain_output = {}
                # print(type(data_sample['edge_index']), isinstance(data_sample['edge_index'], torch_sparse.SparseTensor))
                data_sample.retains_grad = True
                data_sample = data_sample.to(device)

                x = data_sample.x
                if randomise == 1.0:
                    indices = random.sample(range(x.shape[0]), x.shape[0])
                    x = swap_elements(x, range(x.shape[0]), indices)
                    explain_output['swapped_nodes'] = [list(range(x.shape[0])), indices]
                elif randomise != 0.0:
                    sample_len = int(x.shape[0] * randomise // 1)
                    if sample_len % 2 != 0 and sample_len > 1 and x.shape[0] > 1:
                        sample_len -= 1
                    indices = random.sample(range(x.shape[0]), sample_len)
                    # print(indices[:int(sample_len/2)], indices[int(sample_len/2):])
                    x = swap_elements(x, indices[:int(sample_len/2)], indices[int(sample_len/2):])
                    explain_output['swapped_nodes'] = [indices[:int(sample_len/2)], indices[int(sample_len/2):]]
                data_sample.x = x
                edge_index = data_sample.edge_index
                BU_edge_index = data_sample.BU_edge_index
                tweetids = data_sample.tweetids
                explain_output['tweetids'] = tweetids.tolist()

                node_num_to_tweetid = {}
                for node_num, tweetid in enumerate(tweetids):
                    tweetid: torch.Tensor
                    node_num_to_tweetid[int(node_num)] = tweetid.item()

                explain_output['node_num_to_tweetid'] = node_num_to_tweetid

                explain_output['rootindex'] = data_sample.rootindex.item()

                x_sum = torch.sum(x, dim=-1)
                # print('x_sum', x_sum)
                x_sum_top_k = torch.topk(x_sum,
                                         k=x_sum.shape[0])
                # print('x_sum_top_k', x_sum_top_k)
                explain_output['x_sum_top_k'] = [x_sum_top_k.indices.tolist(),
                                                 x_sum_top_k.values.tolist()]

                # TODO: Need to finish the extract method
                if model == 'BiGCN':
                    # TD
                    td_gcn_conv1 = net.TDrumorGCN.conv1
                    td_gcn_conv2 = net.TDrumorGCN.conv2
                    td_gcn_explanations = extract_intermediates_bigcn(td_gcn_conv1, td_gcn_conv2, x,
                                                                      edge_index, data_sample, device)
                    explain_output['td_gcn_explanations'] = td_gcn_explanations
                    # BU
                    bu_gcn_conv1 = net.BUrumorGCN.conv1
                    bu_gcn_conv2 = net.BUrumorGCN.conv2
                    bu_gcn_explanations = extract_intermediates_bigcn(bu_gcn_conv1, bu_gcn_conv2, x,
                                                                      BU_edge_index, data_sample, device)
                    explain_output['bu_gcn_explanations'] = bu_gcn_explanations
                    out_labels = net(data_sample)
                    # _, pred = out_labels.max(dim=-1)
                    # correct = pred.eq(data_sample.y).sum().item()
                    # print(pred.item(), data_sample.y.item(), correct)
                    # explain_output['prediction'] = pred.item()
                    # explain_output['ground_truth'] = data_sample.y.item()
                    # explain_output['correct_prediction'] = correct
                    # raise Exception
                elif model == 'EBGCN':
                    # TD
                    td_gcn_explanations = extract_intermediates_ebgcn(net.TDrumorGCN, x, edge_index, data_sample, device)
                    explain_output['td_gcn_explanations'] = td_gcn_explanations
                    #BU
                    bu_gcn_explanations = extract_intermediates_ebgcn(net.BUrumorGCN, x, BU_edge_index, data_sample, device)
                    explain_output['bu_gcn_explanations'] = bu_gcn_explanations
                    out_labels, _, _ = net(data_sample)
                    # _, pred = out_labels.max(dim=-1)
                    # correct = pred.eq(data_sample.y).sum().item()
                    # print(pred.item(), data_sample.y.item(), correct)
                    # explain_output['prediction'] = pred.item()
                    # explain_output['ground_truth'] = data_sample.y.item()
                    # explain_output['correct_prediction'] = correct
                    # raise Exception
                elif model == 'BERT':
                    x = x.reshape(x.shape[0], -1, 768)
                    if pooling == 'max':
                        new_x = nn.MaxPool1d(256)(x.transpose(2, 1)).squeeze(-1).unsqueeze(0)
                    elif pooling == 'mean':
                        new_x = nn.AvgPool1d(256)(x.transpose(2, 1)).squeeze(-1).unsqueeze(0)
                    bert_out = net.BERT(inputs_embeds=new_x)
                    bert_last_hidden = bert_out.last_hidden_state.squeeze(0)
                    bert_last_hidden_sum = torch.sum(bert_last_hidden, dim=-1)
                    bert_last_hidden_sum_top_k = torch.topk(bert_last_hidden_sum,
                                                            k=bert_last_hidden_sum.shape[0])
                    explain_output['bert_last_hidden_sum_top_k'] = [bert_last_hidden_sum_top_k.indices.tolist(),
                                                                    bert_last_hidden_sum_top_k.values.tolist()]
                    out_labels = net(x)
                _, pred = out_labels.max(dim=-1)
                correct = pred.eq(data_sample.y).sum().item()
                # print(pred.item(), data_sample.y.item(), correct)
                explain_output['prediction'] = pred.item()
                explain_output['ground_truth'] = data_sample.y.item()
                explain_output['correct_prediction'] = correct
                eval_log_string += f'{root_tweetid[0]}: pred: {pred.item()} gt: {data_sample.y.item()}\n'
                conf_mat[data_sample.y.item(), pred.item()] += 1

                # TODO: Finish this
                # extract_gcn_conv(td_gcn_conv1, x, edge_index)
                # raise Exception

                fold_output[int(root_tweetid[0])] = explain_output
                if model == 'BiGCN':
                    if randomise != 0:
                        with open(os.path.join(EXPLAIN_DIR, datasetname, event_name,
                                               f'{root_tweetid[0]}_{model}_r{randomise}_explain2.json'), 'w') as f:
                            json.dump(explain_output, f, indent=1)
                    else:
                        with open(os.path.join(EXPLAIN_DIR, datasetname, event_name,
                                               f'{root_tweetid[0]}_{model}_explain2.json'), 'w') as f:
                            json.dump(explain_output, f, indent=1)
                elif model == 'EBGCN':
                    if randomise != 0:
                        with open(os.path.join(EXPLAIN_DIR, datasetname, event_name,
                                               f'{root_tweetid[0]}_{model}_r{randomise}_explain2.json'), 'w') as f:
                            json.dump(explain_output, f, indent=1)
                    else:
                        with open(os.path.join(EXPLAIN_DIR, datasetname, event_name,
                                               f'{root_tweetid[0]}_{model}_explain2.json'), 'w') as f:
                            json.dump(explain_output, f, indent=1)
                elif model == 'BERT':
                    if randomise != 0:
                        with open(os.path.join(EXPLAIN_DIR, datasetname, event_name,
                                               f'{root_tweetid[0]}_{pooling}{model}_r{randomise}_explain2.json'), 'w') as f:
                            json.dump(explain_output, f, indent=1)
                    else:
                        with open(os.path.join(EXPLAIN_DIR, datasetname, event_name,
                                               f'{root_tweetid[0]}_{pooling}{model}_explain2.json'), 'w') as f:
                            json.dump(explain_output, f, indent=1)
                # if sample_num > 10:
                #     break
                # print(f'Success: {root_tweetid[0]}\n')
            outputs.append(fold_output)
            total_evaluated = conf_mat.sum()
            total_correct = conf_mat.diagonal().sum()
            acc = total_correct/total_evaluated
            eval_log_string += f'Acc: {acc*100:.5f}% [{total_correct}]/[{total_evaluated}]\n'
            for i in range(4):
                precision = conf_mat[i, i] / conf_mat[:, i].sum()
                recall = conf_mat[i, i] / conf_mat[i, :].sum()
                f1 = 2 * precision * recall / (precision + recall)
                eval_log_string += f'Class {i}:\t' \
                                   f'Precision: {precision}\t' \
                                   f'Recall: {recall}\t' \
                                   f'F1: {f1}\n'
            eval_log_string += f' {"":20} | {"":20} | {"Predicted":20}\n' \
                               f' {"":20} | {"":20} | {"Class 0":20} | {"Class 1":20} | {"Class 2":20} | {"Class 3":20}\n'
            for i in range(4):
                if i != 0:
                    eval_log_string += f' {"":20} | {f"Class {i}":20} |'
                else:
                    eval_log_string += f' {"Actual":20} | {f"Class {i}":20} |'
                eval_log_string += f' {conf_mat[i, 0]:20} | {conf_mat[i, 1]:20} |' \
                                   f' {conf_mat[i, 2]:20} | {conf_mat[i, 3]:20}\n'
            with open(evaluation_log_path, 'w') as f:
                f.write(eval_log_string)
            # break
    # with open('explain_outputs.json', 'w') as f:
    #     json.dump(outputs, f, indent=1)
