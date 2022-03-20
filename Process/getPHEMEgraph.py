# -*- coding: utf-8 -*-
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
from transformers import BertTokenizer, BertModel
import torch
import json

cwd = os.getcwd()


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


def constructDataMatrix(tree, tokeniser, model, device):
    tweetids = list(filter(lambda x: x.isnumeric(), tree.keys()))
    id2index = {k: i for i, k in enumerate(tweetids)}
    root_tweetid = tree['root_tweetid']
    root_index = tweetids.index(f'{root_tweetid}')
    row, col = [], []  # sparse matrix representation of adjacency matrix
    texts = []
    label = tree['label']
    for idx, tweetid in enumerate(tweetids):
        # Prep
        texts.append(tree[tweetid]['text'])
        if idx != root_index:
            parent_tweetid = tree[tweetid]['parent_tweetid']
            # print(idx, root_index, parent_tweetid, tweetid)
            child_idx = id2index[f'{tweetid}']
            try:
                parent_idx = id2index[f'{parent_tweetid}']
            except:
                parent_idx = -1
                print(f'Error - Parent tweet ID not found in tree: {tree[f"{tweetid}"]}')
            row.append(parent_idx)
            col.append(child_idx)
    # Batch encode texts with BERT
    encoded_texts = tokeniser(texts,
                             padding='max_length',
                             max_length=256,
                             truncation=True,
                             return_tensors='pt')
    tokens = []
    for text in texts:
        tokens.append(tokeniser.tokenize(text))
    embeddings = model.embeddings(encoded_texts['input_ids'].to(device))
    root_feat = embeddings[root_index].reshape(-1, 256 * 768).cpu().detach().numpy()
    x_word = embeddings.reshape(-1, 256*768).cpu().detach().numpy()
    return x_word, tokens, [row, col], root_feat, root_index, label


def saveTree(tree, tokeniser, model, device):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased').to(device)
    data_matrix = constructDataMatrix(tree, tokeniser, model, device)
    x_word, tokens, edgeindex, root_feat, root_index, label = data_matrix
    tokens = np.array(tokens)
    edgeindex = np.array(edgeindex)
    root_index = np.array(root_index)
    label = np.array(label)
    try:
        np.savez(os.path.join(cwd, 'data', 'PHEMEgraph', f'{tree["root_tweetid"]}.npz'),
                 x=x_word,
                 root=root_feat,
                 edgeindex=edgeindex,
                 rootindex=root_index,
                 y=label,
                 tokens=tokens)
    except:
        try:
            os.makedirs(os.path.join(cwd, 'data', 'PHEMEgraph'))
            print(f"Created graph directory: {os.path.join(cwd, 'data', 'PHEMEgraph')}")
        except:
            pass
    return None

def main():
    event_dir_path = os.path.join(cwd, 'data', 'PHEME')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    # tokeniser = None
    # model = None
    print('loading trees')
    for event_json in filter(lambda x: x.find('.json') != -1, os.listdir(event_dir_path)):
        event_json_path = os.path.join(event_dir_path, event_json)
        with open(event_json_path, 'r') as event_json_file:
            event = json.load(event_json_file)
        # for root_tweetid in event.keys():
        #     tree = event[root_tweetid]
        print('loading dataset')
        event_tweets = list(event.keys())
        Parallel(n_jobs=30, backend='threading')(delayed(saveTree)(event[root_tweetid], tokeniser, model, device) for root_tweetid in tqdm(event_tweets))
    return


if __name__ == '__main__':
    main()