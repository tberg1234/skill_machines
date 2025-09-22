#!/bin/bash
#cd reward_machines
for i in `seq 0 5`; 
do
	(
	# Primitives
	python run.py --alg=smql --env=Office-new-v0 --num_timesteps=100000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --sp data/spql_$i/ --log_path=data/logs/sp-ql/Office-v0/$i

	# # RM6 task -- F(f)
	python run.py --alg=smql --env=Office-new-rm6-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-Ff/$i
	python run.py --alg=smql --env=Office-new-rm6-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --fewshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/fewshot/Office-Ff/$i
   	python run.py --alg=qlearning --env=Office-new-rm6-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm-rs/Office-Ff/$i --use_crm --use_rs
    python run.py --alg=hrm --env=Office-new-rm6-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --save_path=data/models/hrm-rs/Office-Ff/$i/ --log_path=data/logs/hrm-rs/Office-Ff/$i --use_rs

    # # RM7 task -- GF(a) & GF(d)
	python run.py --alg=smql --env=Office-new-rm7-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-GFaANDGFd/$i
	python run.py --alg=smql --env=Office-new-rm7-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --fewshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/fewshot/Office-GFaANDGFd/$i
    python run.py --alg=qlearning --env=Office-new-rm7-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm-rs/Office-GFaANDGFd/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-rm7-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --save_path=data/models/hrm-rs/Office-GFaANDGFd/$i/ --log_path=data/logs/hrm-rs/Office-GFaANDGFd/$i --use_rs

    # # RM8 task -- G(~B) & F(XXn)
	python run.py --alg=smql --env=Office-new-rm8-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-GNOTbANDFXXn/$i
	python run.py --alg=smql --env=Office-new-rm8-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --fewshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/fewshot/Office-GNOTbANDFXXn/$i
    python run.py --alg=qlearning --env=Office-new-rm8-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm-rs/Office-GNOTbANDFXXn/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-rm8-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --save_path=data/models/hrm-rs/Office-GNOTbANDFXXn/$i/ --log_path=data/logs/hrm-rs/Office-GNOTbANDFXXn/$i --use_rs

	) &> log &
done