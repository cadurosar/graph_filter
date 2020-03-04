python test_cifar10.py --model w10 > result_cifar/w10
python test_cifar10.py --model shake > result_cifar/shake
python test_cifar10.py --model pyramid > result_cifar/pyramid

python test_few_shot.py --runs 100000 --dataset cifar > result_fewshot/cifar
python test_few_shot.py --runs 100000 --dataset CUB > result_fewshot/CUB
python test_few_shot.py --runs 100000 --dataset CUB-cross > result_fewshot/CUB-cross
python test_few_shot.py --runs 100000 --dataset miniImagenet > result_fewshot/miniImagenet

