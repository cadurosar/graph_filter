import argparse
import pickle
import torch
import tqdm
import utils

datasets = ["miniImagenet", "CUB", "cifar", "CUB-cross"]
parser = argparse.ArgumentParser(
    description='Denoising DNN features with class low-pass graph filters test on the few-shot task')
parser.add_argument('--dataset', choices=datasets,
                    default=datasets[0], help='Choose the novel dataset to test')
parser.add_argument('--runs', type=int, default=100, help='Number of few-shot iterations')
args = parser.parse_args()




examples_per_class = 5
num_classes = 5
num_queries = 15
path_file = "data/{}.pkl".format(args.dataset)

with open(path_file, 'rb') as f:
    feature_dict = pickle.load(f)

F1 = 1
F2 = 3

filter_ = [1 for i in range(F1)] + [0.2 for i in range(F2-F1)
                                    ] + [0.0 for i in range(examples_per_class-F2)]
filter_ = torch.cuda.FloatTensor(filter_).view(-1, 1)

results_1nn, results_filtered, results_ncm = list(), list(), list()
utils.random.seed(0) #Fix seed for the random generator - Reproducibility
for run in tqdm.tqdm(range(args.runs)):
    train_data, test_data = utils.sample_case(feature_dict, examples_per_class, num_classes, num_queries)
    y_train, y_test = utils.get_labels(num_classes, examples_per_class, num_queries)
    x_train = torch.cuda.FloatTensor(train_data)

    x_test = torch.nn.functional.normalize(torch.cuda.FloatTensor(test_data), dim=1, p=2)

    adj = utils.generate_graphs(
        x_train, k=examples_per_class, examples_per_class=examples_per_class, num_classes=num_classes)
    x_train = torch.nn.functional.normalize(x_train, dim=1, p=2)

    x_train_filtered, y_train_filtered = utils.generate_filtered_features(
        x_train, adj, filter_, num_classes=num_classes, examples_per_class=examples_per_class)
    x_train_filtered = torch.nn.functional.normalize(x_train_filtered, dim=1, p=2)

    results_1nn.append(utils.nearest_neighbor_classifier(
        x_train, x_test, y_train, y_test))
    results_filtered.append(utils.nearest_neighbor_classifier(
        x_train_filtered, x_test, y_train_filtered, y_test))
    results_ncm.append(utils.nearest_mean_classifier(torch.nn.functional.normalize(x_train, dim=1, p=2), torch.cuda.LongTensor(
        y_train), torch.nn.functional.normalize(x_test, dim=1, p=2), torch.cuda.LongTensor(y_test)))

mean_1nn, confiance_1nn = utils.compute_confidence_interval(results_1nn)
mean_filtered, confiance_filtered = utils.compute_confidence_interval(results_filtered)
mean_ncm, confiance_ncm = utils.compute_confidence_interval(results_ncm)


print("Results 1nn {}+-{}, Filtered {}+-{}, NCM {}+-{}".format(
    mean_1nn, confiance_1nn, mean_filtered, confiance_filtered, mean_ncm, confiance_ncm))
