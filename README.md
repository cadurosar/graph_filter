# graph_filter

Code to reproduce the results from the **Denoising DNN features with class low-pass graph filters** paper. 

The files graph_construction.py and graph_utils.py were extracted from the [github of the NNK paper]{https://github.com/STAC-USC/PyNNK_graph_construction} and normalization was extracted from the [github of the SGC paper]{https://github.com/Tiiiger/SGC}. Some functions in util.py were extracted from https://github.com/yhu01/transfer-sgc 


# Reproduce CIFAR-10 Results

```
test_cifar10.py [--model {pyramid,shake,w10}]
```

# Reproduce Few shot results

```
python test_few_shot.py --runs 10000 [--dataset {miniImagenet,CUB,cifar,CUB-cross}]
```
# Reproduce CIFAR-10 all results

```
./test_all.sh
```



# Perform feature extraction for CIFAR-10

Coming soon

# Perform feature extraction for the few shot tasks

We use the readily available features from https://github.com/nupurkmr9/S2M2_fewshot