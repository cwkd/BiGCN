import random
from random import shuffle
import os
import json
import copy

cwd=os.getcwd()


def load9foldData(obj):
    # labelPath = os.path.join(cwd,"data/" +obj+"/"+ obj + "_label_All.txt")
    graph_data_dir_path = os.path.join(cwd, 'data', f'{obj}graph')
    graph_data_check = {tree_id.split('.')[0]: True for tree_id in os.listdir(graph_data_dir_path)}
    data_dir_path = os.path.join(cwd, 'data', obj)
    event_jsons = list(filter(lambda x: x.find('.json') != -1, os.listdir(data_dir_path)))
    # print(event_jsons)
    # labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'nonrumor'], ['false'], ['true'], ['unverified']
    print("loading tree label" )
    # NR,F,T,U = [],[],[],[]
    # l1=l2=l3=l4=0
    # labelDic = {}
    train_folds, test_folds = [], []
    for fold_num, event_json in enumerate(event_jsons):
        event_jsons_copy = copy.copy(event_jsons)
        event_jsons_copy.remove(event_json)
        train_event_ids = []
        for current_event in event_jsons_copy:
            event_json_path = os.path.join(data_dir_path, current_event)
            with open(event_json_path, 'r') as event:
                tweets = json.load(event)
            # print(list(filter(lambda x: graph_data_check.get(x, False), tweets.keys())))
            train_event_ids += list(filter(lambda x: graph_data_check.get(x, False), tweets.keys()))
        train_folds.append(train_event_ids)
        event_json_path = os.path.join(data_dir_path, current_event)
        with open(event_json_path, 'r') as event:
            tweets = json.load(event)
            # print(list(filter(lambda x: graph_data_check.get(x, False), tweets.keys())))
            test_event_ids = list(filter(lambda x: graph_data_check.get(x, False), tweets.keys()))
        test_folds.append(test_event_ids)
    return list(zip(train_folds, test_folds))
