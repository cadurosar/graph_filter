import argparse
import torch
import utils

models = ["pyramid", "shake", "w10"]
parser = argparse.ArgumentParser(
    description='Denoising DNN features with class low-pass graph filters test on the CIFAR-10 dataset')
parser.add_argument('--model', choices=models,
                    default=models[0], help='Choose the feature generating model')
args = parser.parse_args()


x_train, y_train, x_test, y_test = utils.prepare_data_cifar10(args.model)

k = 10
examples_per_class = 5000
num_classes = 10
knn_adj, nnk_adj = utils.generate_graphs(
    x_train, k=k, examples_per_class=examples_per_class, num_classes=num_classes)

F1 = 20
F2 = 55

filter_ = [1 for i in range(F1)] + [0.2 for i in range(F2-F1)
                                    ] + [0.0 for i in range(examples_per_class-F2)]
filter_ = torch.cuda.FloatTensor(filter_).view(-1, 1)

x_train_knn, y_train_knn = utils.generate_filtered_features(
    x_train, knn_adj, filter_, num_classes=num_classes, examples_per_class=examples_per_class)
x_train_nnk, y_train_nnk = utils.generate_filtered_features(
    x_train, nnk_adj, filter_, num_classes=num_classes, examples_per_class=examples_per_class)

result_1nn = utils.nearest_neighbor_classifier(
    x_train, x_test, y_train, y_test)
result_filtered_knn = utils.nearest_neighbor_classifier(
    x_train_knn, x_test, y_train_knn, y_test)
result_filtered_nnk = utils.nearest_neighbor_classifier(
    x_train_nnk, x_test, y_train_nnk, y_test)
result_ncm = utils.nearest_mean_classifier(torch.nn.functional.normalize(x_train, dim=1, p=2), torch.cuda.LongTensor(
    y_train), torch.nn.functional.normalize(x_test, dim=1, p=2), torch.cuda.LongTensor(y_test))

print("Results 1nn {}, KNN {}, NNK {}, NCM {}".format(
    result_1nn, result_filtered_knn, result_filtered_nnk, result_ncm))
