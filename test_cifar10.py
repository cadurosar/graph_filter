import argparse
import torch
import utils

models = ["pyramid", "shake", "w10"]
filters = ["direct", "Simoncelli"]

parser = argparse.ArgumentParser(
    description='Denoising DNN features with class low-pass graph filters test on the CIFAR-10 dataset')
parser.add_argument('--model', choices=models,
                    default=models[0], help='Choose the feature generating model')
parser.add_argument('--filter', choices=filters,
                    default=filters[0], help='Choose the filter we want to apply')
parser.add_argument('--F1', default=20, type=int,
                    help='F1 for the direct filter')
parser.add_argument('--F2', default=55, type=int,
                    help='F2 for the direct filter')
parser.add_argument('--k', default=10, type=int,
                    help='K parameter for both nnk and knn')
parser.add_argument('--alpha', default=0.35, type=float,
                    help='alpha parameter for Simoncelli')

args = parser.parse_args()


x_train, y_train, x_test, y_test = utils.prepare_data_cifar10(args.model)

examples_per_class = 5000
num_classes = 10
knn_adj, nnk_adj = utils.generate_graphs(
    x_train, k=args.k, examples_per_class=examples_per_class, num_classes=num_classes)

if args.filter == "direct":
    alpha = [1 for i in range(args.F1)] + [0.2 for i in range(args.F2-args.F1)
                                          ] + [0.0 for i in range(examples_per_class-args.F2)]
    alpha = torch.cuda.FloatTensor(alpha).view(-1, 1)
else:
    alpha = args.alpha

x_train_knn, y_train_knn = utils.generate_filtered_features(
    x_train, knn_adj, args.filter, alpha, num_classes=num_classes, examples_per_class=examples_per_class)
x_train_nnk, y_train_nnk = utils.generate_filtered_features(
    x_train, nnk_adj, args.filter, alpha, num_classes=num_classes, examples_per_class=examples_per_class)

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
