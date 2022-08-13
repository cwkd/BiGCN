import os
import json
import scipy.stats as stats
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXPLAIN_DIR = os.path.join(DATA_DIR, 'explain')
CENTRALITY_DIR = os.path.join(DATA_DIR, 'centrality')

DATASETNAME = 'PHEME'
EVENT = 'charliehebdo'
FOLD_2_EVENTNAME = {0: 'charliehebdo',
                    1: 'ebola',
                    2: 'ferguson',
                    3: 'germanwings',
                    4: 'gurlitt',
                    5: 'ottawashooting',
                    6: 'prince',
                    7: 'putinmissing',
                    8: 'sydneysiege'}


def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)
    #Find intersection of two sets
    nominator = a.intersection(b)
    #Find union of two sets
    denominator = a.union(b)
    #Take the ratio of sizes
    similarity = len(nominator)/len(denominator)
    # print(a, b, nominator, denominator, similarity)
    return similarity


def compute_metrics(k=10, datasetname=DATASETNAME, event=EVENT, randomise=0.0, errorlog_file_path=None, version=2):
    centrality_subdir = os.path.join(CENTRALITY_DIR, datasetname, event)
    explain_subdir = os.path.join(EXPLAIN_DIR, datasetname, event)
    metrics = [stats.pearsonr, stats.spearmanr, stats.kendalltau, stats.somersd, jaccard_similarity]
    # Change shape if required
    vec = ['out_degree', 'betweenness', 'closeness',
           'bigcn_td_conv1', 'bigcn_td_conv2', 'bigcn_bu_conv1', 'bigcn_bu_conv2',
           'ebgcn_td_conv1', 'ebgcn_td_conv2', 'ebgcn_bu_conv1', 'ebgcn_bu_conv2',
           'maxbert_hidden', 'maxbert_attn', 'meanbert_hidden', 'meanbert_attn']
    metrics_mat = np.zeros((len(os.listdir(centrality_subdir)), len(metrics), len(vec), len(vec)))
    # print(metrics_mat.shape)
    errors = 0
    errors_in_metric = [0, 0, 0, 0, 0]
    error_log = ''
    for tree_num, filename in tqdm(enumerate(sorted(os.listdir(centrality_subdir)))):
        tree_id = filename.split('_')[0]

        if randomise != 0:
            if version == 2:
                centrality_json_path = os.path.join(centrality_subdir, f'{tree_id}_centrality.json')
                bigcn_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_BiGCNv2_r{randomise}_explain3.json')
                ebgcn_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_EBGCNv2_r{randomise}_explain3.json')
                maxBERT_explain_json_path = os.path.join(explain_subdir,
                                                         f'{tree_id}_maxBERT_r{randomise}_explain3.json')
                meanBERT_explain_json_path = os.path.join(explain_subdir,
                                                          f'{tree_id}_meanBERT_r{randomise}_explain3.json')
            else:
                centrality_json_path = os.path.join(centrality_subdir, f'{tree_id}_centrality.json')
                bigcn_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_BiGCN_r{randomise}_explain3.json')
                ebgcn_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_EBGCN_r{randomise}_explain3.json')
                maxBERT_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_maxBERT_r{randomise}_explain3.json')
                meanBERT_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_meanBERT_r{randomise}_explain3.json')
        else:
            if version == 2:
                centrality_json_path = os.path.join(centrality_subdir, f'{tree_id}_centrality.json')
                bigcn_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_BiGCNv2_explain3.json')
                ebgcn_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_EBGCNv2_explain3.json')
                maxBERT_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_maxBERT_explain3.json')
                meanBERT_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_meanBERT_explain3.json')
            else:
                centrality_json_path = os.path.join(centrality_subdir, f'{tree_id}_centrality.json')
                bigcn_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_BiGCN_explain3.json')
                ebgcn_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_EBGCN_explain3.json')
                maxBERT_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_maxBERT_explain3.json')
                meanBERT_explain_json_path = os.path.join(explain_subdir, f'{tree_id}_meanBERT_explain3.json')

        # Centrality
        with open(centrality_json_path, 'r') as f:
            centrality_json = json.load(f)
        out_degree = centrality_json['out_degree'][0]
        if len(out_degree) <= k:
            continue
        out_degree = out_degree[:k] if len(out_degree) > k else out_degree
        closeness = centrality_json['closeness'][0]
        closeness = closeness[:k] if len(closeness) > k else closeness
        betweenness = centrality_json['betweenness'][0]
        betweenness = betweenness[:k] if len(betweenness) > k else betweenness
        # print(out_degree, closeness, betweenness)

        try:
            # BiGCN
            with open(bigcn_explain_json_path, 'r') as f:
                bigcn_explain_json = json.load(f)
            bigcn_td_conv1 = bigcn_explain_json['td_gcn_explanations']['conv1_output_sum_topk'][0]
            bigcn_td_conv1 = bigcn_td_conv1[:k] if len(bigcn_td_conv1) > k else bigcn_td_conv1
            bigcn_td_conv2 = bigcn_explain_json['td_gcn_explanations']['conv2_output_sum_topk'][0]
            bigcn_td_conv2 = bigcn_td_conv2[:k] if len(bigcn_td_conv2) > k else bigcn_td_conv2
            # print(bigcn_td_conv1, bigcn_td_conv2)
            bigcn_bu_conv1 = bigcn_explain_json['bu_gcn_explanations']['conv1_output_sum_topk'][0]
            bigcn_bu_conv1 = bigcn_bu_conv1[:k] if len(bigcn_bu_conv1) > k else bigcn_bu_conv1
            bigcn_bu_conv2 = bigcn_explain_json['bu_gcn_explanations']['conv2_output_sum_topk'][0]
            bigcn_bu_conv2 = bigcn_bu_conv2[:k] if len(bigcn_bu_conv2) > k else bigcn_bu_conv2
            # print(bigcn_bu_conv1, bigcn_bu_conv2)

            # EBGCN
            with open(ebgcn_explain_json_path, 'r') as f:
                ebgcn_explain_json = json.load(f)
            ebgcn_td_conv1 = ebgcn_explain_json['td_gcn_explanations']['conv1_output_sum_topk'][0]
            ebgcn_td_conv1 = ebgcn_td_conv1[:k] if len(ebgcn_td_conv1) > k else ebgcn_td_conv1
            ebgcn_td_conv2 = ebgcn_explain_json['td_gcn_explanations']['conv2_output_sum_topk'][0]
            ebgcn_td_conv2 = ebgcn_td_conv2[:k] if len(ebgcn_td_conv2) > k else ebgcn_td_conv2
            # print(ebgcn_td_conv1, ebgcn_td_conv2)
            ebgcn_bu_conv1 = ebgcn_explain_json['bu_gcn_explanations']['conv1_output_sum_topk'][0]
            ebgcn_bu_conv1 = ebgcn_bu_conv1[:k] if len(ebgcn_bu_conv1) > k else ebgcn_bu_conv1
            ebgcn_bu_conv2 = ebgcn_explain_json['bu_gcn_explanations']['conv2_output_sum_topk'][0]
            ebgcn_bu_conv2 = ebgcn_bu_conv2[:k] if len(ebgcn_bu_conv2) > k else ebgcn_bu_conv2
            # print(ebgcn_bu_conv1, ebgcn_bu_conv2)

            with open(maxBERT_explain_json_path, 'r') as f:
                maxBERT_explain_json = json.load(f)
            maxbert_hidden = maxBERT_explain_json['bert_last_hidden_sum_top_k'][0]
            maxbert_hidden = maxbert_hidden[:k] if len(maxbert_hidden) > k else maxbert_hidden
            maxbert_attn = maxBERT_explain_json['bert_attentions_top_k'][0]
            maxbert_attn = maxbert_attn[:k] if len(maxbert_attn) > k else maxbert_attn
            with open(meanBERT_explain_json_path, 'r') as f:
                meanBERT_explain_json = json.load(f)
            meanbert_hidden = meanBERT_explain_json['bert_last_hidden_sum_top_k'][0]
            meanbert_hidden = meanbert_hidden[:k] if len(meanbert_hidden) > k else meanbert_hidden
            meanbert_attn = meanBERT_explain_json['bert_attentions_top_k'][0]
            meanbert_attn = meanbert_attn[:k] if len(meanbert_attn) > k else meanbert_attn

        except:
            print(f'Error: Missing explanation files for Tree num {tree_id}\tRandomise: {randomise}')
            continue
        vec = [out_degree, betweenness, closeness,
               bigcn_td_conv1, bigcn_td_conv2, bigcn_bu_conv1, bigcn_bu_conv2,
               ebgcn_td_conv1, ebgcn_td_conv2, ebgcn_bu_conv1, ebgcn_bu_conv2,
               maxbert_hidden, maxbert_attn, meanbert_hidden, meanbert_attn]
        # print(tree_id)
        temp_mat = np.zeros((len(metrics), len(vec), len(vec)))
        for metric_num in range(len(metrics) - 1):
            for i in range(len(vec)):
                for j in range(len(vec)):
                    try:
                        temp_mat[metric_num, i, j] = metrics[metric_num](vec[i], vec[j])[0]
                    except:
                        # print(metrics[metric_num](vec[i], vec[j]))
                        try:
                            temp_mat[metric_num, i, j] = metrics[metric_num](vec[i], vec[j]).statistic
                        except:
                            # print(tree_id)
                            # print(i, j, vec[i], vec[j])
                            error_log += f'{tree_id}, {metric_num}, {i}, {j}, {vec[i]}, {vec[j]}\n'
                            errors_in_metric[metric_num] += 1
                            errors += 1
                            continue
            # print(f'{metric_num}\n', metrics_mat[tree_num, metric_num])
        else:
            # print(metrics[metric_num + 1])
            for i in range(len(vec)):
                for j in range(len(vec)):
                    temp_mat[metric_num + 1, i, j] = metrics[metric_num + 1](vec[i], vec[j])
            # print(f'{metric_num + 1}\n', metrics_mat[tree_num, metric_num + 1])
        metrics_mat[tree_num] = temp_mat
        # print(vec)
    for metric_num in range(len(metrics)):
        print(f'Event: {event_name}\tMetric: {metrics[metric_num]}\tError: {errors_in_metric[metric_num]}')
    print(f'Event: {event_name}\tTotal Errors: {errors}')
    with open(errorlog_file_path, 'w') as f:
        f.write(error_log)
    # np.save(os.path.join(DATA_DIR, f'{datasetname}_{event}'), metrics_mat)
    if version == 2:
        np.save(os.path.join(EXPLAIN_DIR, f'k{k}_{datasetname}_{event}_r{randomise}_3'), metrics_mat)
    else:
        np.save(os.path.join(EXPLAIN_DIR, f'k{k}_{datasetname}_{event}_r{randomise}_3v2'), metrics_mat)
    return metrics_mat


def summarise_metrics(mat, save_path):
    print(f'Save path: {save_path}')
    metrics = [stats.pearsonr, stats.spearmanr, stats.kendalltau, stats.somersd, jaccard_similarity]
    vec = ['out_degree', 'betweenness', 'closeness',
           'bigcn_td_conv1', 'bigcn_td_conv2', 'bigcn_bu_conv1', 'bigcn_bu_conv2',
           'ebgcn_td_conv1', 'ebgcn_td_conv2', 'ebgcn_bu_conv1', 'ebgcn_bu_conv2',
           'maxbert_hidden', 'maxbert_attn', 'meanbert_hidden', 'meanbert_attn']
    s = ''
    avg_metrics_mat = np.zeros((len(metrics), len(vec), len(vec)))
    for metric_num in range(len(metrics)):
        s += f'{metrics[metric_num]}\n'
        s += f'{"":20}' + '\t|\t'
        for i in range(len(vec)):
            s += f'{vec[i]:20}' + '\t|\t'
        s += '\n'
        for i in range(len(vec)):
            s += f'{vec[i]:20}' + '\t|\t'
            for j in range(len(vec)):
                # print(np.sum(mat[:, metric_num, i, j]), np.count_nonzero(mat[:, metric_num, i, j]))
                avg = np.sum(mat[:, metric_num, i, j]) / np.count_nonzero(mat[:, metric_num, i, j])
                s += f'{avg:20}' + '\t|\t'
                avg_metrics_mat[metric_num, i, j] = avg
            else:
                s += '\n'
        s += '\n'
    with open(save_path, 'w') as f:
        f.write(s)
    print(f'Metrics saved to : {save_path}')
    return avg_metrics_mat


if __name__ == '__main__':
    # metrics_mat = compute_metrics()
    randomise_types = [1.0, 0.5, 0.25, 0.0]
    k_types = [10, 5]
    vec = ['out_degree', 'betweenness', 'closeness',
           'bigcn_td_conv1', 'bigcn_td_conv2', 'bigcn_bu_conv1', 'bigcn_bu_conv2',
           'ebgcn_td_conv1', 'ebgcn_td_conv2', 'ebgcn_bu_conv1', 'ebgcn_bu_conv2',
           'maxbert_hidden', 'maxbert_attn', 'meanbert_hidden', 'meanbert_attn']
    num_metrics = 5
    version = 2
    for k in k_types:
        for randomise in randomise_types:
            all_event_metrics_mat = np.zeros((len(FOLD_2_EVENTNAME), num_metrics, len(vec), len(vec)))
            for fold_num in range(len(FOLD_2_EVENTNAME)):
                event_name = FOLD_2_EVENTNAME[fold_num]
                print(f'Processing k{k}_{DATASETNAME}_{event_name}_r{randomise}\n')
                if version == 2:
                    errorlog_file_path = os.path.join(
                        EXPLAIN_DIR,
                        f'k{k}_{DATASETNAME}_{event_name}_r{randomise}_compare_similarity_error_log3v2.txt')
                else:
                    errorlog_file_path = os.path.join(
                        EXPLAIN_DIR,
                        f'k{k}_{DATASETNAME}_{event_name}_r{randomise}_compare_similarity_error_log3.txt')
                metrics_mat = compute_metrics(k, DATASETNAME, event_name, randomise, errorlog_file_path, version)
                if version == 2:
                    summary_file_path = os.path.join(
                        EXPLAIN_DIR,
                        f'k{k}_{DATASETNAME}_{event_name}_r{randomise}_metrics_summary3v2.txt')
                else:
                    summary_file_path = os.path.join(
                        EXPLAIN_DIR,
                        f'k{k}_{DATASETNAME}_{event_name}_r{randomise}_metrics_summary3.txt')
                all_event_metrics_mat[fold_num] = summarise_metrics(metrics_mat, save_path=summary_file_path)
            else:
                if version == 2:
                    all_event_summary_file_path = os.path.join(
                        EXPLAIN_DIR,
                        f'k{k}_{DATASETNAME}_allevents_r{randomise}_metrics_summary3v2.txt')
                else:
                    all_event_summary_file_path = os.path.join(
                        EXPLAIN_DIR,
                        f'k{k}_{DATASETNAME}_allevents_r{randomise}_metrics_summary3.txt')
                all_event_metrics_mat.sum(axis=0)
                summarise_metrics(all_event_metrics_mat, save_path=all_event_summary_file_path)
    # metrics_mat = np.load('data/PHEME_charliehebdo.npy')
    # metrics_mat = np.load('data/PHEME_charliehebdo_bothBERT.npy')
    # metrics_mat = np.load('data/PHEME_charliehebdo_bothgcn_bothBERT.npy')
    # summarise_metrics(metrics_mat, save_path='metrics_summary_bothgcn_bothBERT.txt')