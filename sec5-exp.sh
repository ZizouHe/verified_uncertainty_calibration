#!/bin/bash

python sec5-exp.py --top --num_bins 15 --pow=2
python sec5-exp.py --top --num_bins 100 --pow=2
python sec5-exp.py --top --num_bins 15 --pow=1
python sec5-exp.py --top --num_bins 100 --pow=1

python sec5-exp.py --num_bins 15 --pow=2
python sec5-exp.py --num_bins 100 --pow=2
python sec5-exp.py --num_bins 15 --pow=1
python sec5-exp.py --num_bins 100 --pow=1

python sec5-exp.py --top --num_bins 15 --pow=2 --data_path='../data/imagenet_probs.dat' --prefix='../pic/imagenet_debiased_'
python sec5-exp.py --top --num_bins 100 --pow=2 --data_path='../data/imagenet_probs.dat' --prefix='../pic/imagenet_debiased_'
python sec5-exp.py --top --num_bins 15 --pow=1 --data_path='../data/imagenet_probs.dat' --prefix='../pic/imagenet_debiased_'
python sec5-exp.py --top --num_bins 100 --pow=1 --data_path='../data/imagenet_probs.dat' --prefix='../pic/imagenet_debiased_'
