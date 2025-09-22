#!/bin/bash

python enjoy.py --algo ql --env Office-new-v0 --sp_dir data/sp-ql/Office-v0/0/

	python run.py --alg=smql --env=Office-new-rm6-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-Ff/$i
