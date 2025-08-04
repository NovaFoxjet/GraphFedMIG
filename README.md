# GraphFedMIG

> **GraphFedMIG: Tackling Class Imbalance in Federated Graph Learning via Mutual Information-Guided Generation**

## Abstract

Federated graph learning (FGL) enables multiple clients to collaboratively train powerful graph neural networks (GNNs) without sharing their private, decentralized graph data. Inherited from generic federated learning, FGL is critically challenged by data heterogeneity, where non-IID data distributions across clients can severely impair model performance. A particularly destructive form of this is class imbalance, which causes the global model to become biased towards majority classes and fail at identifying rare but critical events. This issue is exacerbated in FGL, as nodes from a minority class are often surrounded by biased neighborhood information, hindering the learning of expressive embeddings. To grapple with this challenge, we propose GraphFedMIG, a novel FGL framework that reframes the problem as a federated generative data augmentation task. GraphFedMIG employs a hierarchical generative adversarial network where each client trains a local generator to synthesize high-fidelity feature representations. To provide tailored supervision, clients are grouped into clusters, each sharing a dedicated discriminator. Crucially, the framework designs a mutual information-guided mechanism to steer the evolution of these client generators. By calculating each client's unique informational value, this mechanism corrects the local generator parameters, ensuring that subsequent rounds of mutual information-guided generation are focused on producing high-value, minority-class features. We conduct extensive experiments on four real-world datasets, and the results validate the superiority of GraphFedMIG compared with other state-of-the-art baselines.

## File guidance

### partition

The dataset index for each client participating in the training process.

### data.py

Obtain the dataset from the link and process the data.

### GraphFedMIG_client.py

The client-side implementation of GraphFedMIG, specifically including the generator training process and the mutual information maximization loss.

### GraphFedMIG_main.py

The training implementation of GraphFedMIG, where the model is configured and trained here.

### GraphFedMIG_server.py

The implementation of GraphFedMIG's server and client clusters, including mutual information parameter aggregation, hierarchical clustering, and discriminator training.

### model.py

It includes all the models needed during the training process.

### utils.py

It includes the utility classes that will be used during the training process.