#!/bin/bash
python sec3-exp.py --data_path="../data/cifar_probs_vgg16.dat" --fig_name="../pic/cifar_vgg16_l2.png" --n1=1000 --n2=1000 --pow=2
python sec3-exp.py --data_path="../data/cifar_probs_vgg16.dat" --fig_name="../pic/cifar_vgg16_l1.png" --n1=1000 --n2=1000 --pow=1
python sec3-exp.py --data_path="../data/cifar_probs_resnet18.dat" --fig_name="../pic/cifar_resnet18_l2.png" --n1=1000 --n2=1000 --pow=2
python sec3-exp.py --data_path="../data/cifar_probs_resnet18.dat" --fig_name="../pic/cifar_resnet18_l1.png" --n1=1000 --n2=1000 --pow=1
python sec3-exp.py --data_path="../data/cifar_probs_densenet121.dat" --fig_name="../pic/cifar_densenet121_l2.png" --n1=1000 --n2=1000 --pow=2
python sec3-exp.py --data_path="../data/cifar_probs_densenet121.dat" --fig_name="../pic/cifar_densenet121_l1.png" --n1=1000 --n2=1000 --pow=1
python sec3-exp.py --data_path="../data/imagenet_probs.dat" --fig_name="../pic/imagenet_l2_ext_1510.png" --n1=15000 --n2=10000 --pow=2 --bins_list="2,4,8,16,32,64,128,256,512,1024,2048,4096"
python sec3-exp.py --data_path="../data/imagenet_probs.dat" --fig_name="../pic/imagenet_l1_ext_1510.png" --n1=15000 --n2=10000 --pow=1 --bins_list="2,4,8,16,32,64,128,256,512,1024,2048,4096"
