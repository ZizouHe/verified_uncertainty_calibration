#!/bin/bash

python sec4-exp.py --data_path='../data/cifar_probs_vgg16.dat' --n1=1000 --ce_fig='../pic/cifar_vgg16_top_ce.png' --mse_fig='../pic/cifar_vgg16_top_mse.png' --times=100 --cifar --top
python sec4-exp.py --data_path='../data/cifar_probs_resnet18.dat' --n1=1000 --ce_fig='../pic/cifar_resnet18_top_ce.png' --mse_fig='../pic/cifar_resnet18_top_mse.png' --times=100 --cifar --top
python sec4-exp.py --data_path='../data/cifar_probs_densenet121.dat' --n1=1000 --ce_fig='../pic/cifar_densenet121_top_ce.png' --mse_fig='../pic/cifar_densenet121_top_mse.png' --times=100 --cifar --top

python sec4-exp.py --data_path='../data/cifar_probs_vgg16.dat' --n1=1000 --ce_fig='../pic/cifar_vgg16_margin_ce.png' --mse_fig='../pic/cifar_vgg16_margin_mse.png' --times=100 --cifar
python sec4-exp.py --data_path='../data/cifar_probs_resnet18.dat' --n1=1000 --ce_fig='../pic/cifar_resnet18_margin_ce.png' --mse_fig='../pic/cifar_resnet18_margin_mse.png' --times=100 --cifar
python sec4-exp.py --data_path='../data/cifar_probs_densenet121.dat' --n1=1000 --ce_fig='../pic/cifar_densenet121_margin_ce.png' --mse_fig='../pic/cifar_densenet121_margin_mse.png' --times=100 --cifar

python sec4-exp.py --data_path='../data/imagenet_probs.dat' --n1=1000 --ce_fig='../pic/imagenet_top_ce.png' --mse_fig='../pic/imagenet_top_mse.png' --times=100 --top
python sec4-exp.py --data_path='../data/imagenet_probs.dat' --n1=25000 --ce_fig='../pic/imagenet_margin_ce.png' --mse_fig='../pic/imagenet_margin_mse.png' --times=100

python sec4-exp.py --data_path='../data/cifar_probs_vgg16.dat' --n1=1000 --ce_fig='../pic/cifar_vgg16_top_ce_ext.png' --times=100 --cifar --top --exp=2
python sec4-exp.py --data_path='../data/cifar_probs_vgg16.dat' --n1=1000 --ce_fig='../pic/cifar_vgg16_margin_ce_ext.png' --times=100 --cifar --exp=2
python sec4-exp.py --data_path='../data/cifar_probs_resnet18.dat' --n1=1000 --ce_fig='../pic/cifar_resnet18_top_ce_ext.png' --times=100 --cifar --top --exp=2
python sec4-exp.py --data_path='../data/cifar_probs_resnet18.dat' --n1=1000 --ce_fig='../pic/cifar_resnet18_margin_ce_ext.png' --times=100 --cifar --exp=2
python sec4-exp.py --data_path='../data/cifar_probs_densenet121.dat' --n1=1000 --ce_fig='../pic/cifar_densenet121_top_ce_ext.png' --times=100 --cifar --top --exp=2
python sec4-exp.py --data_path='../data/cifar_probs_densenet121.dat' --n1=1000 --ce_fig='../pic/cifar_densenet121_margin_ce_ext.png' --times=100 --cifar --exp=2
