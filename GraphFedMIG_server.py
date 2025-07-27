import copy

from operator import index

import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

from model import MLP, GCN, SGC, SAGE,Generator,Discriminator

class GraphFedMIGServer:
    def __init__(self, client_list, client_ids, trainIdx, dataset, hidden, device, pretrain_epochs, threshold):
        self.client_list = client_list
        self.client_ids = client_ids
        self.trainIdx = trainIdx
        self.dataset = dataset
        self.hidden = hidden
        self.device = device
        self.pretrain_epochs = pretrain_epochs
        self.threshold = threshold
        self.num_train_nodes = [len(self.trainIdx[client_id]) for client_id in self.client_ids]
        self.global_train_loss_history = []
        self.global_test_acc_history = []
        self.client_proto=[]

        self.pretrain_clients()

        self.client_features = [client.compute_average_features() for client in self.client_list]
        self.class_num=len(client_list[0].classes)
        self.clusters = [[i] for i in range(len(self.client_list))]
        self.dis = []
        self.disop = []
        self.pavg = []


        self.criterion=nn.CrossEntropyLoss()

    def pretrain_clients(self):
        for client in self.client_list:
            client.pretrain(self.pretrain_epochs)

    def compute_cosine_similarity(self, feature1, feature2):
        return cosine_similarity(feature1.cpu().numpy().reshape(1, -1), feature2.cpu().numpy().reshape(1, -1))[0][0]

    def hierarchical_clustering(self):
        while len(self.clusters) > 1:
            max_sim = -1
            cluster1_idx = -1
            cluster2_idx = -1

            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    cluster1_feature = torch.mean(torch.stack([self.client_features[c] for c in self.clusters[i]]),
                                                  dim=0)
                    cluster2_feature = torch.mean(torch.stack([self.client_features[c] for c in self.clusters[j]]),
                                                  dim=0)

                    sim = self.compute_cosine_similarity(cluster1_feature, cluster2_feature)
                    if sim > max_sim:
                        max_sim = sim
                        cluster1_idx = i
                        cluster2_idx = j

            if max_sim < self.threshold:
                break

            self.clusters[cluster1_idx].extend(self.clusters[cluster2_idx])
            del self.clusters[cluster2_idx]

    def train(self, rounds, rounds2=5, lr=0.01):
        # 执行层次聚类
        self.hierarchical_clustering()
        self.get_all_proto()
        for cluster in self.clusters:
            self.dis.append(Discriminator(in_channel=self.class_num,hidden=self.class_num).to(self.device))
        for cluster in self.clusters:
            n=0
            self.disop.append(torch.optim.Adam(self.dis[n].parameters(), lr=lr))
            n=n+1
        best_val_loss = 1e6
        best_test_acc = 0
        # 初始化优化器
        optimizer = torch.optim.Adam(self.client_list[0].gnn.parameters(), lr=lr)

        # 多轮训练
        for round in range(1, rounds + 1):
            # 簇内训练
            for cluster in self.clusters:
                index=0
                self.intra_cluster_training(cluster, rounds2,index)
                index=index+1


            self.inter_cluster_training()

            optimizer.zero_grad()
            for client in self.client_list:
                output, _ = client.gnn(client.features, client.edge_index)
                loss = F.cross_entropy(output[client.trainIdx], client.labels[client.trainIdx])
                loss.backward()
            optimizer.step()

            loss_list = []
            val_loss_list = []
            num_val_list = []
            num_test_list = []
            correct_train_list = []
            correct_val_list = []
            correct_test_list = []
            for i, client in enumerate(self.client_list):
                loss, val_loss, num_val, num_test, correct_train, correct_val, correct_test = client.stats()
                loss_list.append(client.train_loss_history[-1])
                val_loss_list.append(val_loss)
                num_val_list.append(num_val)

                num_test_list.append(num_test)
                correct_train_list.append(correct_train)
                correct_val_list.append(correct_val)
                correct_test_list.append(correct_test)

            total_val = np.sum(num_val_list)
            total_test = np.sum(num_test_list)
            train_loss = np.sum(loss_list) / np.sum(self.num_train_nodes)
            val_loss = np.sum(val_loss_list) / total_val
            acc_train = np.sum(correct_train_list) / np.sum(self.num_train_nodes)
            acc_val = np.sum(correct_val_list) / total_val
            acc_test = np.sum(correct_test_list) / total_test
            self.global_train_loss_history.append(train_loss)
            self.global_test_acc_history.append(acc_test)
            print(
                'Round: {:4d} | train_loss: {:9.5f} | val_loss: {:9.5f} | acc_train: {:7.5f} | acc_val: {:7.5f} | acc_test: {:7.5f}'
                .format(round, train_loss, val_loss, acc_train, acc_val, acc_test))

            if 1==1:
                best_val_loss = val_loss
                best_test_acc = acc_test

                num_minority_list = []
                correct_minority_list = []
                recall_list = []
                tm_list = []
                precision_list = []
                for i, client in enumerate(self.client_list):
                    correct_minority, num_minority, avg_recall, avg_precision = client.print_count_nodes_per_class()
                    num_minority_list.append(num_minority)
                    correct_minority_list.append(correct_minority)
                    tm_list.append(correct_minority / num_minority)
                    recall_list.append(avg_recall)
                    precision_list.append(avg_precision)
                true_acc_minority = np.mean(tm_list)
                acc_minority = np.sum(correct_minority_list) / np.sum(num_minority_list)
                allavg_recall = np.mean(recall_list)
                allavg_precision = np.mean(precision_list)

        return best_test_acc, true_acc_minority, acc_minority, allavg_recall, allavg_precision

    def intra_cluster_training(self, cluster, rounds,index1):
        index=index1
        for _ in range(1):


            fake_p=[]
            true_p=[]
            output, _ = self.client_list[cluster[0]].gnn(self.client_list[cluster[0]].features, self.client_list[cluster[0]].edge_index)





            for c in cluster:

                output, _ = self.client_list[c].gnn(self.client_list[c].features, self.client_list[c].edge_index)
                fake_p.append(output[self.client_list[c].trainIdx])



                n=0

                temp= self.add_gaussian_noise(self.client_list[c].proto[self.client_list[c].labels[0]].unsqueeze(0))

                for x in range(len(self.client_list[c].labels)):
                    if n!=0:

                        temp_tensor=self.add_gaussian_noise(self.client_list[c].proto[self.client_list[c].labels[x]].unsqueeze(0))
                        temp =torch.cat((temp, temp_tensor), dim=0)
                    n=n+1


                temp2, _  =self.client_list[c].gnn(temp, self.client_list[c].edge_index)
                true_p.append(temp2[self.client_list[c].trainIdx])

            n = 0
            temp2=fake_p[0]
            for c in cluster:
                if n !=0:
                    temp2 = torch.cat((temp2, fake_p[n]), dim=0)
                n=n+1
            pavg1 = F.softmax((temp2).float(), dim=1)
            self.pavg.append(pavg1)





            for _ in range(rounds):
                n = 0
                fake_output = self.dis[index](fake_p[0])
                real_output= torch.softmax(true_p[0],dim=0)
                #print(fake_output[0])
                for c in cluster:
                    if n !=0:
                        fake_output = torch.cat((fake_output, self.dis[index](fake_p[n])), dim=0)
                        real_output = torch.cat((real_output, torch.softmax(true_p[n],dim=0)), dim=0)

                    n=n+1
                d_loss = self.criterion(fake_output, real_output)

                self.disop[index].zero_grad()
                d_loss.backward(retain_graph=True)
                self.disop[index].step()

            temp3= self.dis[index](fake_p[0])
            all_p_avg= temp3[0]
            y=0
            for i in self.client_list[cluster[0]].trainIdx:
                if y!=0:
                    all_p_avg=all_p_avg+temp3[y]
                y=y+1
            all_p_avg=all_p_avg/len(self.client_list[cluster[0]].trainIdx)

            z=0
            for c in cluster:
                temp4=self.dis[index](fake_p[z])

                if z!=0:
                    r=0
                    for i in self.client_list[c].trainIdx:
                        all_p_avg=all_p_avg+temp4[r]
                        r=r+1

                    all_p_avg=all_p_avg/len(self.client_list[c].trainIdx)
                z=z+1
            mi_client=[]
            for c in cluster:
                output, _ = self.client_list[c].gnn(self.client_list[c].features, self.client_list[c].edge_index)
                d_output = self.dis[index](output)
                mi=0
                for i in self.client_list[c].trainIdx:
                    temp=metrics.mutual_info_score(output[i].detach().numpy(),
                                                                  d_output[i].detach().numpy()),
                    mi=mi+temp[0]
                mi = mi/len(self.client_list[c].trainIdx)
                mi_client.append(mi)
            avgMi=0
            for i in range(len(mi_client)):
                avgMi=avgMi+mi_client[i]

            avgMi=avgMi/len(mi_client)

            client_models = [self.client_list[c].gnn.state_dict() for c in cluster]
            global_model = self.client_list[cluster[0]].gnn
            n=0

            for c in cluster:
                if n==0:
                    for key in global_model.state_dict():
                        if global_model.state_dict()[key].data.dtype == torch.float32:
                            global_model.state_dict()[key].data.copy_(
                                (mi_client[n]/avgMi)*client_models[n][key].float()
                            )
                else:
                    for key in global_model.state_dict():
                        if global_model.state_dict()[key].data.dtype == torch.float32:
                            global_model.state_dict()[key].data.copy_(
                                global_model.state_dict()[key].float()+(mi_client[n]/avgMi)*client_models[n][key].float()
                            )

                n=n+1
            for key in global_model.state_dict():
                if global_model.state_dict()[key].data.dtype == torch.float32:
                    global_model.state_dict()[key].data.copy_(
                        global_model.state_dict()[key].float()/len(cluster)
                    )



            for c in cluster:
                self.client_list[c].gnn.load_state_dict(global_model.state_dict())
                self.client_list[c].local_update(self.client_list[c].gnn,len(cluster),copy.deepcopy(self.dis[index]), all_p_avg)


    def inter_cluster_training(self):
        # 将每个簇视为一个客户端，收集所有簇的判别模型参数
        go_dis_model = self.dis[0].state_dict()
        for key in go_dis_model:
            if go_dis_model[key].data.dtype == torch.float32:
                go_dis_model[key].data.copy_(
                    torch.mean(torch.stack([self.dis[index].state_dict()[key].float() for index in range(len(self.clusters))]),
                               dim=0)
                )


    def get_all_proto(self):
        for m in range(len(self.client_list)):
            real_p = []
            for n in self.client_list[m].trainIdx:
                for j in self.client_list[m].classes:
                    if self.client_list[m].labels[n] == j:
                        real_p.append(self.client_list[m].proto[j])
            self.client_proto.append(real_p)

    def add_gaussian_noise(self,tensor, mean=0, std=0.0001):
        noise = torch.randn(tensor.size()) * std + mean

        noisy_tensor = tensor + noise
        return noisy_tensor
