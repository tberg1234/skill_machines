#!/bin/bash
#cd reward_machines
for i in `seq 0 60`; 
do
	(
	# RM1 task
	python run.py --alg=smql --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --zeroshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/zeroshot/Office-Coffee-Task-v0/$i
    python run.py --alg=smql --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --fewshot --sp data/spql_$i/ --log_path=data/logs/sm-ql/fewshot/Office-Coffee-Task-v0/$i
	# python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql/Office-Coffee-Task-v0/$i
	# python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/rm-ql-rs/Office-Coffee-Task-v0/$i --use_rs
	# python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm/Office-Coffee-Task-v0/$i --use_crm
	# python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/crm-rs/Office-Coffee-Task-v0/$i --use_crm --use_rs
	# python run.py --alg=hrm --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm/Office-Coffee-Task-v0/$i
	# python run.py --alg=hrm --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=data/logs/hrm-rs/Office-Coffee-Task-v0/$i --use_rs
    )&> log &
done
