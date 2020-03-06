time python test_cifar10.py --model w10 --filter Simoncelli --alpha 0.05 > result_cifar/w10
time python test_cifar10.py --model shake --filter Simoncelli --alpha 0.05 > result_cifar/shake
time python test_cifar10.py --model pyramid --filter Simoncelli --alpha 0.05 > result_cifar/pyramid

python test_few_shot.py --runs 100000 --dataset cifar > result_fewshot/cifar
python test_few_shot.py --runs 100000 --dataset CUB > result_fewshot/CUB
python test_few_shot.py --runs 100000 --dataset CUB-cross > result_fewshot/CUB-cross
python test_few_shot.py --runs 100000 --dataset miniImagenet > result_fewshot/miniImagenet

