import random
import torch
import numpy as np
import tqdm
import scipy
import graph_utils
import graph_construction
import normalization


def nearest_mean_classifier(train_set, train_labels, test_set, test_labels):
    # Compute the means of the feature vectors of the same classes
    n_way = torch.max(train_labels) + 1
    means = torch.zeros(n_way, train_set.shape[1]).cuda()
    for label in range(n_way):
        means[label] = torch.mean(train_set[train_labels == label], dim=0)
    means = torch.nn.functional.normalize(means, dim=1, p=2)
    similarities = torch.mm(test_set, means.T)
    # Choose the labels according to the closest mean
    predicted = torch.argmax(similarities, dim=1)
    # Compute accuracy
    total = test_labels.shape[0]
    correct = (predicted == test_labels).sum()
    test_acc = (100 - 100. * correct / total).item()
    return np.round(test_acc, 2)


def nearest_neighbor_classifier(x_train, x_test, y_train, y_test):
    sim = torch.matmul(torch.nn.functional.normalize(x_test, dim=1, p=2),
                       torch.nn.functional.normalize(x_train, dim=1, p=2).T).cpu().numpy()
    ranks = np.argmax(sim, axis=1)
    result = 1-((y_train[ranks] == y_test).sum())/y_test.shape[0]
    result = np.round(100*result, 2)
    return result


def prepare_data_autoaugment(model):
    x_train = torch.cuda.FloatTensor(
        np.load("data/train_{}.npy".format(model)))

    beta = np.load("data/{}_test.npz".format(model), allow_pickle=True)
    testloader = zip(beta["x"], beta["y"])

    y_trains = []
    for a in range(10):
        y_trains.append(np.zeros(5000, dtype=np.int32)+a)
    y_train = np.concatenate(y_trains)

    x_test = list()
    y_test = list()
    for inputs, targets in tqdm.tqdm(testloader):
        with torch.no_grad():
            y_test.append(targets)
            outputs = torch.FloatTensor(inputs)
            x_test.append(outputs)
    x_test = torch.cuda.FloatTensor(np.concatenate(x_test))
    y_test = np.concatenate(y_test)
    return x_train, y_train, x_test, y_test


def prepare_data_w10():
    alfa = np.load("data/w10_train.npz", allow_pickle=True)
    beta = np.load("data/w10_test.npz", allow_pickle=True)
    trainloader = zip(alfa["x"], alfa["y"])
    testloader = zip(beta["x"], beta["y"])

    y_trains = []
    for a in range(10):
        y_trains.append(np.zeros(5000, dtype=np.int32)+a)
    y_train = np.concatenate(y_trains)

    x_train = list()
    y_train = list()
    for inputs, targets in tqdm.tqdm(trainloader):
        with torch.no_grad():
            y_train.append(targets)
            outputs = torch.FloatTensor(inputs)
            x_train.append(outputs)
    x_train = torch.cat(x_train).view(50000, -1)
    y_train = np.array(y_train).astype(np.int32)

    x_test = list()
    y_test = list()
    for inputs, targets in tqdm.tqdm(testloader):
        with torch.no_grad():
            y_test.append(targets)
            outputs = torch.FloatTensor(inputs)
            x_test.append(outputs)
    x_test = torch.cat(x_test).view(10000, -1)
    y_test = np.array(y_test).astype(np.int32)
    return x_train.cuda(), y_train, x_test.cuda(), y_test


def prepare_data_cifar10(model):
    if model == "w10":
        return prepare_data_w10()
    else:
        return prepare_data_autoaugment(model)


def generate_graphs(x_train, k=10, sigma=1e-6, examples_per_class=5000, num_classes=10):
    knn_adj = scipy.sparse.lil_matrix((x_train.shape[0], x_train.shape[0]))
    nnk_adj = scipy.sparse.lil_matrix((x_train.shape[0], x_train.shape[0]))
    adj = scipy.sparse.lil_matrix((x_train.shape[0], x_train.shape[0]))
    for a in range(num_classes):

        local_x_train = torch.nn.functional.normalize(
            x_train[a*examples_per_class:(a+1)*examples_per_class], dim=1, p=2)
        G = torch.mm(local_x_train, local_x_train.T).cpu().numpy()
        if k < examples_per_class:
            knn_mask = graph_utils.create_directed_KNN_mask(
                D=G, knn_param=k, D_type='similarity')
            knn_adj[a*examples_per_class:(a+1)*examples_per_class, a*examples_per_class:(a+1) *
                    examples_per_class] = graph_construction.knn_graph(G, knn_mask, k, sigma)
            nnk_adj[a*examples_per_class:(a+1)*examples_per_class, a*examples_per_class:(a+1) *
                    examples_per_class] = graph_construction.nnk_graph(G, knn_mask, k, sigma)
        else:
            adj[a*examples_per_class:(a+1)*examples_per_class, a*examples_per_class:(a+1) *
                examples_per_class] = G
    if k < examples_per_class:
        return knn_adj, nnk_adj
    else:
        return adj

def low_pass_filter_direct(x, adj, filter_):

    nodes_to_keep = np.where(adj.sum(axis=1) > 0)[0]
    adj = adj.todense()
    np.fill_diagonal(adj, 0)
    real_adj = adj[nodes_to_keep]
    real_adj = real_adj[:, nodes_to_keep]
    x = x[nodes_to_keep]

    laplacian = torch.cuda.FloatTensor(
        normalization.normalized_laplacian(real_adj).todense())
    _, eigenVectors = torch.symeig(laplacian, eigenvectors=True)
    x = torch.matmul(eigenVectors.T, x)
    x = filter_[:len(nodes_to_keep)]*x
    x = torch.matmul(eigenVectors, x)
    return x


def generate_filtered_features(x_train, adj, filter_, examples_per_class=5000, num_classes=10):
    filtered_x_train = list()
    y_train = list()
    for a in range(num_classes):
        local_x_train = torch.nn.functional.normalize(
            x_train[a*examples_per_class:(a+1)*examples_per_class], dim=1, p=2)
        W = adj[a*examples_per_class:(a+1)*examples_per_class,
                a*examples_per_class:(a+1)*examples_per_class]
        filtered_x = low_pass_filter_direct(local_x_train, W, filter_)
        filtered_x_train.append(filtered_x)
        y_train.append(np.zeros(filtered_x.shape[0])+a)
    filtered_x_train = torch.cat(filtered_x_train)
    y_train = np.concatenate(y_train)
    return filtered_x_train, y_train

#from https://github.com/yhu01/transfer-sgc
def sample_case(ld_dict, shot, n_ways, n_queries):
    sample_class = random.sample(list(ld_dict.keys()), n_ways)
    train_input = []
    test_input = []
    for each_class in sample_class:
        samples = random.sample(ld_dict[each_class], shot + n_queries)
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input

#from https://github.com/yhu01/transfer-sgc
def get_labels(num_ways, shot, num_queries):
    train_labels = []
    test_labels = []
    classes = [i for i in range(num_ways)]
    for each_class in classes:
        train_labels += [each_class] * shot
        test_labels += [each_class] * num_queries

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return train_labels, test_labels

#from https://github.com/yhu01/transfer-sgc
def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return np.round(m, 1), np.round(pm, 2)
