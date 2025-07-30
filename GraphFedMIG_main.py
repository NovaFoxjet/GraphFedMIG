import torch
import numpy as np

from GraphFedMIG_client import GraphFedMIGClient
from GraphFedMIG_server import GraphFedMIGServer
import os
import numpy as np
import torch
import math

from data import load_data
import matplotlib.pyplot as plt

def run(dataset_name, epochs,round, hidden, lr, pretrain_epochs, threshold, seed):
    arch_name = os.path.basename(__file__).split('.')[0]
    data_path = './data/'
    file_names = sorted(os.listdir('./partition/'+f'{dataset_name}/'))

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    dataset, num_clients, trainIdx, valIdx, testIdx = load_data(dataset_name, data_path, file_names)

    # 初始化参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建客户端
    clients = []
    for i in range(num_clients):
        client = GraphFedMIGClient(i, dataset, trainIdx[i], valIdx[i], testIdx[i], lr, epochs, hidden, device,dataset_name)
        clients.append(client)

    # 创建服务器
    server = GraphFedMIGServer(clients, list(range(num_clients)), trainIdx, dataset, hidden, device, pretrain_epochs, threshold)

    # 开始训练
    best_test_acc, true_acc_minority,acc_minority,allavg_recall, allavg_precision = server.train(rounds=round)
    # 绘制训练损失曲线


    print('Arch: {:s} | dataset: {:s} | lr: {:6.4f} | epochs:{:2d} | hidden: {:3d} | pretrain_epochs: {:2d} | threshold: {:4.2f} | seed: {:2d} | best_test_acc: {:6.4f} | true_acc_minority: {:6.4f}| acc_minority: {:6.4f}| allavg_recall: {:6.4f}| allavg_precision: {:6.4f}'
          .format(arch_name, dataset_name, lr, epochs, hidden, pretrain_epochs, threshold, seed, best_test_acc,true_acc_minority,acc_minority,allavg_recall, allavg_precision ))


if __name__ == '__main__':
    dataset_name = 'EllipticBitcoinDataset'
    epochs = 5
    round = 100
    hidden = 64
    lr = 0.01
    pretrain_epochs = 20
    threshold = -100
    seed = 0
    run(dataset_name, epochs,round, hidden, lr, pretrain_epochs, threshold, seed)