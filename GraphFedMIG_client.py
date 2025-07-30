from itertools import count

import torch
import numpy as np
import torch.nn.functional as F
import math

from model import SAGE,Generator
from utils import subgraph
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
class GraphFedMIGClient:
    def __init__(self, client_id, dataset, trainIdx, valIdx, testIdx, lr, epochs, hidden, device,dataset_name):
        self.client_id = client_id
        self.node_list = trainIdx + valIdx + testIdx
        self.data = dataset[0]
        self.train_loss_history = []
        self.test_acc_history = []
        self.trainIdx = list(range(0, len(trainIdx)))
        self.valIdx = list(range(len(trainIdx), len(trainIdx) + len(valIdx)))
        self.testIdx = list(range(len(trainIdx) + len(valIdx), len(trainIdx) + len(valIdx) + len(testIdx)))
        self.features = self.data.x[self.node_list]
        self.labels = self.data.y[self.node_list]
        self.features = self.features.to(device)
        self.labels = self.labels.squeeze().to(device)
        if dataset_name == "EllipticBitcoinDataset":
            self.labels = torch.tensor(self.labels.squeeze().to(device) / 2, dtype=torch.int64)
        self.label_dist = torch.bincount(self.labels[self.trainIdx]).tolist()
        self.classes = [c for c in range(dataset.num_classes)]
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.proto = self.getproto()
        self.gnn = Generator(in_channel=dataset.num_node_features, out_channel=dataset.num_classes, hidden=hidden).to(device)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)

        self.subgraph = subgraph(subset=torch.tensor(self.node_list, dtype=torch.long), edge_index=self.data.edge_index,
                                 relabel_nodes=True,  num_nodes=self.data.num_nodes)
        self.edge_index = self.subgraph[0].to(device)


    def compute_average_features(self):
        return torch.mean(self.features, dim=0)

    def pretrain(self, pretrain_epochs):
        self.gnn.train()
        for epoch in range(1, pretrain_epochs + 1):
            self.optimizer.zero_grad()
            output, _ = self.gnn(self.features, self.edge_index)
            loss = F.cross_entropy(output[self.trainIdx], self.labels[self.trainIdx])
            loss.backward()
            self.optimizer.step()
            self.train_loss_history.append(loss.item())
        return self.gnn.state_dict()

    def get_test_accuracy(self):
        self.gnn.eval()
        with torch.no_grad():
            output, _ = self.gnn(self.features, self.edge_index)
            pred = output[self.testIdx].argmax(dim=1)
            correct = (pred == self.labels[self.testIdx]).sum().item()
            acc = correct / len(self.testIdx)
            self.test_acc_history.append(acc)
            return acc

    def local_update(self, model,h1,go_model,p_gavg1):

        self.gnn.load_state_dict(model.state_dict())
        self.gnn.train()
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            output, _ = self.gnn(self.features, self.edge_index)
            p_gavg=p_gavg1.detach()
            h=h1
            temp=torch.softmax(torch.mean(output[self.trainIdx],dim=0),dim=0)

            p_avg = ((temp+p_gavg)/2).unsqueeze(0)

            p_d = F.log_softmax(torch.mean(output[self.trainIdx],dim=0), dim=0).unsqueeze(0)
            kl = torch.nn.KLDivLoss(reduction="batchmean")
            p_gavg2=F.log_softmax(p_gavg, dim=0).unsqueeze(0)
            part1=kl(p_d.t(),p_avg.detach().t())
            part2=kl(p_gavg2.t(),p_avg.detach().t())
            part3=-(h+1)*math.log(h+1)+h*math.log(h)


            gan_loss = part1+h*part2+part3
            mi_loss3 = torch.tensor(metrics.mutual_info_score(output[self.trainIdx].argmax(dim=1).detach().numpy(),
                                                        self.labels[self.trainIdx].detach().numpy()),requires_grad=True)
            mi_loss2 = - np.log2(np.exp(1)) * (torch.mean(output[self.trainIdx]) - torch.log(
                torch.mean(torch.exp(self.labels[self.trainIdx]))))

            loss = F.cross_entropy(output[self.trainIdx], self.labels[self.trainIdx])+0.0001 * gan_loss+0.0000001*mi_loss2

            loss.backward()
            self.optimizer.step()
            self.train_loss_history.append(loss.item())
        return self.gnn.state_dict()

    def stats(self):
        self.gnn.eval()
        with torch.no_grad():
            output, _ = self.gnn(self.features, self.edge_index)
            loss = F.cross_entropy(output[self.trainIdx], self.labels[self.trainIdx]).item()
            val_loss = F.cross_entropy(output[self.valIdx], self.labels[self.valIdx]).item()
            num_val = len(self.valIdx)
            num_test = len(self.testIdx)
            correct_train = (output[self.trainIdx].argmax(dim=1) == self.labels[self.trainIdx]).sum().item()
            correct_val = (output[self.valIdx].argmax(dim=1) == self.labels[self.valIdx]).sum().item()
            correct_test = (output[self.testIdx].argmax(dim=1) == self.labels[self.testIdx]).sum().item()
        return loss, val_loss, num_val, num_test, correct_train, correct_val, correct_test

    def getproto(self):
        tempall=[]
        all=torch.tensor(torch.zeros(self.features[0].size()), dtype=torch.float)
        for n in self.trainIdx:
            all=all+self.features[n]
        all=all/len(self.trainIdx)

        for j in self.classes:
            count=0
            temp=torch.tensor(torch.zeros(self.features[0].size()), dtype=torch.float)
            for n in self.trainIdx:
                if self.labels[n]==j:
                    count=count+1
                    temp=self.features[n]+temp
            if count==0:
                tempall.append(all)
            else:
                tempall.append(temp/count)

        return tempall

    def print_count_nodes_per_class(self, gnn=None):
        # self.gnn.load_state_dict(gnn.state_dict())
        self.gnn.eval()
        output, emb = self.gnn(self.features, self.edge_index)
        prediction = output.argmax(dim=1)[self.testIdx]
        labels = self.labels[self.testIdx]
        majority_class = np.argmax(self.label_dist)
        right_count = []
        all_count = []
        all_right_count = []
        for i in range(0, len(self.label_dist)):
            tempcount = 0
            tempcount2 = 0
            tempcount3 = 0
            for j in range(0, len(self.testIdx)):
                if labels[j] == i and prediction[j] == i:
                    tempcount = tempcount + 1
                if labels[j] == i:
                    tempcount2 = tempcount2 + 1
                if prediction[j] == i:
                    tempcount3 = tempcount3 + 1
            right_count.append(tempcount)
            all_count.append(tempcount2)
            all_right_count.append(tempcount3)

        avg_recall = 0
        avg_precision = 0
        tlen = 0
        tlen2 = 0
        for i in range(0, len(self.label_dist)):
            if all_count[i] != 0 and right_count[i]!=0:
                tlen = tlen + 1
                avg_recall = avg_recall + (right_count[i] / all_count[i])
            if all_right_count[i] != 0 and right_count[i]!=0:
                tlen2 = tlen2 + 1
                avg_precision = avg_precision + (right_count[i] / all_right_count[i])

        avg_recall = avg_recall / tlen
        avg_precision = avg_precision / tlen2
        label_counter = len(torch.where(labels != majority_class)[0])
        idx = torch.where(labels != majority_class)[0]
        correct = torch.sum(labels[idx] == prediction[idx])
        acc = correct / label_counter
        return correct.item(), label_counter, avg_recall, avg_precision

    def add_gaussian_noise(self,tensor, mean=0, std=0.0001):
        noise = torch.randn(tensor.size()) * std + mean

        noisy_tensor = tensor + noise
        return noisy_tensor


