import pandas as pd
from tqdm import tqdm
import argparse
import pickle
from collections import *
import random
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
from model import *

name2idx = {'icustay_id': 0, 'Heart_Rate': 1, 'Respiratory_Rate': 2, 'Systolic_BP': 3, 'Diastolic_BP': 4, 'SPO2': 5,
            'MAP': 6, 'age': 7, 'is_male': 8, 'score': 9}


def split_labels(data_set):
    data, labels = [], []
    for x in data_set:
        data.append(x[1:-1])
        # data.append([x[1], x[2], [3], x[5]])
        labels.append(x[-1])
        # labels.append(1 if x[-1] == 3 else 0)
    max_d = [max(d[i] for d in data) for i in range(len(data[0]))]
    min_d = [min(d[i] for d in data) for i in range(len(data[0]))]
    new_data = []
    for ds in data:
        t = []
        for i, d in enumerate(ds):
            t.append((d - min_d[i]) / (max_d[i] - min_d[i]))
        new_data.append(t)
    data = new_data
    # c = Counter(labels)
    # print(c)
    return data, labels


def build_model(args):
    device = f'cuda:{args.gpu_id}' if (torch.cuda.is_available() and args.gpu_id is not None) else 'cpu'
    
    def get_model(_args):
        if _args.model == "NN":
            return LinearNNModel(hidden_size=_args.hidden_size, lr=_args.learning_rate, class_num=_args.class_num,
                                 weight_decay=_args.weight_decay, batch_size=_args.batch_size, weights=args.weights,
                                 max_patience=_args.max_patience, embedding_size=_args.embedding_size, device=device)
        elif _args.model == 'LR':
            return LRModel()
        elif _args.model == 'DT':
            return DTModel()
        elif _args.model == 'SVM':
            return SVMModel()
        elif _args.model == 'KNN':
            return KNNModel(args.k)
        elif _args.model == 'RF':
            return RFModel()
        elif _args.model == 'BST':
            return BSTModel()
        elif _args.model == 'BAG':
            return BAGModel()
        elif _args.model == 'GBDT':
            return GBDTModel()
        else:
            raise AssertionError('Invalid model name!')

    if args.model.endswith('++'):
        args.class_num = 2
        args.weights = [1, 2]
        args.model = args.model[:-2]
        model = lambda: get_model(args)
        return PlusPlusModel(model)
    else:
        return get_model(args)


def evaluate(predict_labels, test_labels):
    acc = accuracy_score(test_labels, predict_labels)
    p = precision_score(test_labels, predict_labels, average='macro')
    recall = recall_score(test_labels, predict_labels, average='macro')
    f1 = f1_score(test_labels, predict_labels, average='macro')
    report = classification_report(test_labels, predict_labels, digits=4)
    return acc, p, recall, f1, report


def train_and_eval(args, train_set, test_set):
    train_data, train_labels = split_labels(train_set)
    args.embedding_size = len(train_data[0])
    model = build_model(args)
    model.train_model(train_data, train_labels)

    test_data, test_labels = split_labels(test_set)
    predict_label = model.predict(test_data)
    return evaluate(predict_label, test_labels)


def pre_process(path, _delete=False):
    data = pd.read_excel(path)
    print(type(data.columns.values.tolist()))
    print(data.columns.values)
    print(">>>before:", len(data))
    processed_data = []
    data.fillna("", inplace=True)
    if _delete:
        for line in tqdm(data.values.tolist()):  # 获取行号的索引，并对其进行遍历：
            # 根据i来获取每一行指定的数据 并利用to_dict转成字典
            exist_null = False
            for item in data.columns.values:
                if line[name2idx[item]] == "":
                    exist_null = True
                    break
                line[name2idx[item]] = float(line[name2idx[item]])
            if exist_null:
                continue
            if (line[name2idx['is_male']] == 0 or line[name2idx['is_male']] == 1) and \
                    (30 <= line[name2idx['Heart_Rate']] <= 260) and \
                    (10 <= line[name2idx['MAP']] <= 200) and \
                    (5 <= line[name2idx['Respiratory_Rate']] <= 70) and \
                    (40 <= line[name2idx['Systolic_BP']] <= 300) and \
                    (40 <= line[name2idx['Diastolic_BP']] <= 300) and \
                    (0 <= line[name2idx['SPO2']] <= 100) and \
                    (line[name2idx['score']] == 0 or line[name2idx['score']] == 1 or line[name2idx['score']] == 2
                     or line[name2idx['score']] == 3 or line[name2idx['score']] == 4):
                processed_data.append(line)
    else:
        records = [{} for _ in range(5)]
        # record of average values
        for line in data.values.tolist():  # 获取行号的索引，并对其进行遍历：
            if line[-1] == "" or (line[-1] != 0 and line[-1] != 1 and line[-1] != 2 and line[-1] != 3 and line[-1] != 4):
                continue
            line[-1] = int(line[-1])

            if line[name2idx['is_male']] != "" and line[name2idx['is_male']] != 0 and line[name2idx['is_male']] != 1:
                continue
            if line[name2idx['Heart_Rate']] != "" and (
                    line[name2idx['Heart_Rate']] < 30 or line[name2idx['Heart_Rate']] > 260):
                continue
            if line[name2idx['MAP']] != "" and (
                    line[name2idx['MAP']] < 10 or line[name2idx['MAP']] > 200):
                continue
            if line[name2idx['Respiratory_Rate']] != "" and (
                    line[name2idx['Respiratory_Rate']] < 5 or line[name2idx['Respiratory_Rate']] > 70):
                continue
            if line[name2idx['Systolic_BP']] != "" and (
                    line[name2idx['Systolic_BP']] < 40 or line[name2idx['Systolic_BP']] > 300):
                continue
            if line[name2idx['Diastolic_BP']] != "" and (
                    line[name2idx['Diastolic_BP']] < 40 or line[name2idx['Diastolic_BP']] > 300):
                continue
            if line[name2idx['SPO2']] != "" and (line[name2idx['SPO2']] <= 0 or line[name2idx['SPO2']] > 100):
                continue
            # 根据i来获取每一行指定的数据 并利用to_dict转成字典
            for item in data.columns.values:
                if name2idx[item] != 0 and name2idx[item] != 9 and line[name2idx[item]] != "":
                    line[name2idx[item]] = float(line[name2idx[item]])
                    t = dict(records[line[-1]]).get(item, (0.0, 0))
                    records[line[-1]][item] = (t[0] + line[name2idx[item]], t[1] + 1)
            processed_data.append(line)
        for score in records:
            for item in score:
                score[item] = score[item][0] / score[item][1]
        count_complete = 0
        for line in processed_data:
            is_null = False
            for item in data.columns.values:
                if line[name2idx[item]] == "":
                    is_null = True
                    print(line)
                    line[name2idx[item]] = records[int(line[-1])][item]
                    print(f"{item}: {line[name2idx[item]]}\n")
            if is_null:
                count_complete += 1
        print(">>>complete:", count_complete)
    print(">>>after:", len(processed_data))
    with open('data/data.pkl', 'wb') as out:
        pickle.dump(processed_data, out)


def get_split_data_file(data, radio=None):
    if radio is None:
        radio = [6, 3, 1]
    train_data = []
    valid_data = []
    test_data = []
    total = sum(radio)
    for i, d in enumerate(radio):
        radio[i] = float(d) / total
        radio[i] += radio[i - 1] if i > 0 else 0
    for line in data:
        t = random.random()
        if t < radio[0]:
            train_data.append(line)
        elif t < radio[1]:
            valid_data.append(line)
        else:
            test_data.append(line)
    with open('data/train_data.pkl', 'wb') as out:
        pickle.dump(train_data, out)
    with open('data/valid_data.pkl', 'wb') as out:
        pickle.dump(valid_data, out)
    with open('data/test_data.pkl', 'wb') as out:
        pickle.dump(test_data, out)


def main(args):
    # pre_process('data/train_set.xlsx', _delete=True)
    # return
    # read data
    with open('data/data.pkl', 'rb') as fin:
        data = pickle.load(fin)
    # get_split_data_file(data)

    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    acc_list, p_list, r_list, f1_list = [], [], [], []
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        train_set = [data[i] for i in train_idx]
        test_set = [data[i] for i in test_idx]
        acc, p, r, f1, report = train_and_eval(args, train_set, test_set)
        acc_list.append(acc)
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)
        print('-' * 80)
        print(f'fold {i}')
        print('-' * 80)
        print(report)
        print('-' * 80)
        if not args.k_fold:
            return
    print('Average Acc:', np.array(acc_list).mean())
    print('Average P:', np.array(p_list).mean())
    print('Average R:', np.array(r_list).mean())
    print('Average F1:', np.array(f1_list).mean())


def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser for Respiratory Score')
    configs_group = parser.add_argument_group(title='Running Configs')
    configs_group.add_argument('--model', type=str, help='the name of model')
    configs_group.add_argument('--k-fold', action='store_true', help='whether to use KFold')
    configs_group.add_argument('--k', type=int, help='the number of k in kNN', default=5)
    configs_group.add_argument('--gpu-id', type=int, help='the gpu', default=None)

    model_group = parser.add_argument_group(title='Model parameters')
    model_group.add_argument('--class-num', type=int, help='the number of class', default=5)
    model_group.add_argument('--embedding-size', type=int, help='the embedding size of word', default=8)
    model_group.add_argument('--hidden-size', type=int, help='the hidden size of LSTM', default=8)

    training_group = parser.add_argument_group(title='Training parameters')
    training_group.add_argument('--learning-rate', type=float, help='the learning rate', default=0.001)
    training_group.add_argument('--weight-decay', type=float, help='the weight decay for AdamOptimizer', default=0.0005)
    training_group.add_argument('--weights', type=list, help='the weights for classes', default=None)
    training_group.add_argument('--batch-size', type=int, help='the batch size', default=1024)
    training_group.add_argument('--max-patience', type=int, help='the max patience for early stopping', default=5)
    training_group.add_argument('--valid-rate', type=float, help='the rate to split data for validation', default=0.2)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
