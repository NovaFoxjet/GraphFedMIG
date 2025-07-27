import numpy as np
from torch_geometric.datasets import EllipticBitcoinDataset,Twitch,FacebookPagePage,Actor

def load_data(dataset_name, data_path, file_names):
    if dataset_name == "EllipticBitcoinDataset":
        dataset = EllipticBitcoinDataset(root=f'{data_path}/{dataset_name}')
    elif dataset_name == "Twitch":
        dataset = Twitch(root=f'{data_path}/{dataset_name}', name="DE")
    elif dataset_name == "FacebookPagePage":
        dataset = FacebookPagePage(root=f'{data_path}/{dataset_name}')
    elif dataset_name == "Actor":
        dataset = Actor(root=f'{data_path}/{dataset_name}')
    num_nodes = 0

    trainIdx = []
    valIdx = []
    testIdx = []
    file_list = []
    num_clients = 0
    for file in file_names:
        if file.find(f'{dataset_name}') == 0:
            file_list.append(file)
            num_clients += 1
    np.random.shuffle(file_list)
    for file in file_list:
        node_list = np.loadtxt(f'./partition/{dataset_name}/{file}').astype(int)
        np.random.shuffle(node_list)
        trainIdx.append(list(node_list)[: int(0.4 * len(node_list))])
        valIdx.append(list(node_list)[int(0.4 * len(node_list)): int(0.7 * len(node_list))])
        testIdx.append(list(node_list)[int(0.7 * len(node_list)):])
        num_nodes += len(node_list)

    return dataset, num_clients, trainIdx, valIdx, testIdx
