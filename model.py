from copy import deepcopy
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn


class RespiratoryScoreModel:

    def __init__(self):
        super(RespiratoryScoreModel, self).__init__()

    def train_model(self, data, scores):
        pass

    def predict(self, data):
        pass


class LinearNNModel(RespiratoryScoreModel, nn.Module):

    def __init__(self, lr, weight_decay, batch_size, max_patience, hidden_size, embedding_size, device, class_num=5,
                 weights=None):
        super(LinearNNModel, self).__init__()
        if weights is None:
            weights = [9, 3, 2, 1, 3]
        self.embedding_size = embedding_size
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.weights = weights
        self.batch_size = batch_size
        self.max_patience = max_patience
        self.hidden_size = hidden_size
        self.best_model_params = None
        self.valid_rate = 0.2
        self.class_num = class_num
        # self.linear = nn.Linear(self.embedding_size, self.class_num)
        self.dropout = 0.1
        self.sizes = [
            self.embedding_size,
            *(self.hidden_size for _ in range(37)),
            self.class_num]
        self.module_list = nn.ModuleList()
        for i in range(0, len(self.sizes) - 2):
            self.module_list.append(nn.Sequential(
                nn.Linear(self.sizes[i], self.sizes[i + 1]),
                nn.Tanh(),
                nn.Dropout(self.dropout)
            ))
        self.module_list.append(nn.Linear(self.sizes[-2], self.sizes[-1]))
        self.n_report = 100
        self.to(self.device)

    def forward(self, data):
        output = self.module_list[0](data)
        for module in self.module_list[1:-1]:
            data = module(output)
            output = torch.relu(output + data)
        return self.module_list[-1](output)

    def fit(self, data, scores):
        return self.train_model(data, scores)

    def train_model(self, data, scores):
        all_train_data, all_valid_data, all_train_scores, all_valid_scores = \
            train_test_split(data, scores, test_size=self.valid_rate, random_state=0)

        # weights = torch.tensor([18, 6, 3, 2, 6]).float().cuda()
        # weights = torch.tensor([1.4, 1]).float().cuda()
        loss_func = nn.CrossEntropyLoss(weight=torch.tensor(self.weights).float().cuda()).to(self.device)
        # loss_func = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        best_f1 = 0
        best_loss = float('inf')
        best_model_params = None
        best_epoch = 0
        patience = 0

        i = 0
        while True:
            loss_item = 0
            # shuffle train_texts, train_scores
            # all_items = list(zip(all_train_data, all_train_scores))
            # random.shuffle(all_items)
            # all_train_data, all_train_scores = zip(*all_items)
            all_train_data_tensor = torch.tensor(all_train_data).to(self.device)
            for train_texts, train_scores in self.generate_batches(all_train_data_tensor, all_train_scores):
                outputs = self.forward(train_texts)
                train_scores_tensor = torch.tensor(train_scores).long().to(self.device)
                loss = loss_func(outputs, train_scores_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item += loss.item()

            all_valid_pred = self.predict(all_valid_data)
            f1 = self.valid_model(all_valid_pred, all_valid_scores)
            progress = False
            if f1 > best_f1:
                # if loss_item < best_loss:
                progress = True
                patience = 0
                best_f1 = f1
                best_loss = loss_item
                best_epoch = i + 1
                best_model_params = deepcopy(self.state_dict())
            else:
                patience += 1
                if patience > self.max_patience:
                    # print('Early stopped')
                    break
            if (i + 1) % self.n_report == 0:
                print(f'[Epoch {i + 1}] [Loss: {loss_item}] [F1: {f1}] {"*" if progress else ""}')
            i += 1
        print(f'Epoch {best_epoch} gets the best valid f1 {best_f1}')
        if best_model_params is None:
            print('Model not learned')
            return
        self.best_model_params = best_model_params
        self.load_state_dict(self.best_model_params)

    @staticmethod
    def valid_model(pred, scores):
        return f1_score(scores, pred, average='macro')

    def predict(self, data):
        data = torch.tensor(data).to(self.device)
        self.eval()
        results = []
        with torch.no_grad():
            for texts in self.generate_batches(data):
                outputs = self.forward(texts)
                pred = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                results.extend(pred)
        self.train()
        return results

    def generate_batches(self, *args):
        n_data = len(args[0])
        for i in range(0, n_data, self.batch_size):
            if len(args) == 1:
                yield args[0][i:i + self.batch_size]
            else:
                yield [arg[i:i + self.batch_size] for arg in args]


class LRModel(RespiratoryScoreModel):

    def __init__(self):
        super(LRModel, self).__init__()
        self.clf = LogisticRegression()

    def train_model(self, data, scores):
        self.clf.fit(data, scores)

    def predict(self, data):
        return self.clf.predict(data)


class RFModel(RespiratoryScoreModel):

    def __init__(self):
        super(RFModel, self).__init__()
        self.clf = RandomForestClassifier()

    def train_model(self, data, scores):
        self.clf.fit(data, scores)

    def predict(self, data):
        return self.clf.predict(data)


class BSTModel(RespiratoryScoreModel):

    def __init__(self):
        super(BSTModel, self).__init__()
        self.clf = AdaBoostClassifier()

    def train_model(self, data, scores):
        self.clf.fit(data, scores)

    def predict(self, data):
        return self.clf.predict(data)


class BAGModel(RespiratoryScoreModel):

    def __init__(self):
        super(BAGModel, self).__init__()
        self.clf = BaggingClassifier()

    def train_model(self, data, scores):
        self.clf.fit(data, scores)

    def predict(self, data):
        return self.clf.predict(data)


class GBDTModel(RespiratoryScoreModel):

    def __init__(self):
        super(GBDTModel, self).__init__()
        self.clf = GradientBoostingClassifier()

    def train_model(self, data, scores):
        self.clf.fit(data, scores)

    def predict(self, data):
        return self.clf.predict(data)


class DTModel(RespiratoryScoreModel):

    def __init__(self):
        super(DTModel, self).__init__()
        self.clf = DecisionTreeClassifier()

    def train_model(self, data, scores):
        self.clf.fit(data, scores)

    def predict(self, data):
        return self.clf.predict(data)


class SVMModel(RespiratoryScoreModel):

    def __init__(self):
        super(SVMModel, self).__init__()
        self.clf = SVC()

    def train_model(self, data, scores):
        self.clf.fit(data, scores)

    def predict(self, data):
        return self.clf.predict(data)


class KNNModel(RespiratoryScoreModel):

    def __init__(self, k):
        super(KNNModel, self).__init__()
        self.clf = KNeighborsClassifier(k)

    def train_model(self, data, scores):
        self.clf.fit(data, scores)

    def predict(self, data):
        return self.clf.predict(data)


class RFPPModel(RespiratoryScoreModel):

    def __init__(self):
        super(RFPPModel, self).__init__()
        self.clf_3 = RandomForestClassifier()
        self.clf = RandomForestClassifier()
        self.clf_2 = RandomForestClassifier()
        # self.clf_4 = RandomForestClassifier()
        # self.clf_1 = RandomForestClassifier()

    def train_model(self, data, scores):
        ds = list(zip(data, scores))
        ds3 = [(d, 1 if s == 3 else 0) for d, s in ds]
        ds2 = [(d, 1 if s == 2 else 0) for d, s in ds if s != 3]
        ds = [(d, s) for d, s in ds if s != 3 and s != 2]
        # ds4 = [(d, 1 if s == 4 else 0) for d, s in ds if s != 3 and s != 2]
        # ds1 = [(d, 1 if s == 1 else 0) for d, s in ds if s != 3 and s != 2 and s != 4]
        self.clf_3.fit(*zip(*ds3))
        self.clf.fit(*zip(*ds))
        self.clf_2.fit(*zip(*ds2))
        # self.clf_4.fit(*zip(*ds4))
        # self.clf_1.fit(*zip(*ds1))

    def predict(self, data):
        s3 = self.clf_3.predict(data)
        s = self.clf.predict(data)
        s2 = self.clf_2.predict(data)
        # s4 = self.clf_4.predict(data)
        # s1 = self.clf_1.predict(data)
        # return s3 * 3 + (1 - s3) * (s2 * 2 + (1 - s2) * (s4 * 4 + (1 - s4) * (s1 * 1)))
        return s3 * 3 + (1 - s3) * (s2 * 2 + (1 - s2) * s)


class PlusPlusModel(RespiratoryScoreModel):

    def __init__(self, classifier, train_name="train_model", predict_name="predict"):
        super(PlusPlusModel, self).__init__()
        self.train_name = train_name
        self.predict_name = predict_name
        self.clf_3 = classifier()
        self.clf_2 = classifier()
        self.clf_4 = classifier()
        self.clf_1 = classifier()

    def train_model(self, data, scores):
        ds = list(zip(data, scores))
        ds3 = [(d, 1 if s == 3 else 0) for d, s in ds]
        ds2 = [(d, 1 if s == 2 else 0) for d, s in ds if s != 3]
        ds4 = [(d, 1 if s == 4 else 0) for d, s in ds if s != 3 and s != 2]
        ds1 = [(d, 1 if s == 1 else 0) for d, s in ds if s != 3 and s != 2 and s != 4]
        print("classifier 3 is training...")
        self.clf_3.__getattribute__(self.train_name)(*zip(*ds3))
        print("classifier 2 is training...")
        self.clf_2.__getattribute__(self.train_name)(*zip(*ds2))
        print("classifier 4 is training...")
        self.clf_4.__getattribute__(self.train_name)(*zip(*ds4))
        print("classifier 1 is training...")
        self.clf_1.__getattribute__(self.train_name)(*zip(*ds1))

    def predict(self, data):
        s3 = np.array(self.clf_3.__getattribute__(self.predict_name)(data))
        s2 = np.array(self.clf_2.__getattribute__(self.predict_name)(data))
        s4 = np.array(self.clf_4.__getattribute__(self.predict_name)(data))
        s1 = np.array(self.clf_1.__getattribute__(self.predict_name)(data))
        return s3 * 3 + (1 - s3) * (s2 * 2 + (1 - s2) * (s4 * 4 + (1 - s4) * (s1 * 1)))
