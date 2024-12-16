#!/bin/bash
#cd reward_machines
for i in `seq 0 25`; 
do
	(
	# RM1 task
	python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql/Office-Coffee-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql-rs/Office-Coffee-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm/Office-Coffee-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm-rs/Office-Coffee-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm/Office-Coffee-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm-rs/Office-Coffee-Task-v0/$i --use_rs

	# RM3 task
	python run.py --alg=qlearning --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql/Office-CoffeeMail-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql-rs/Office-CoffeeMail-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm/Office-CoffeeMail-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm-rs/Office-CoffeeMail-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm/Office-CoffeeMail-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-rm3-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm-rs/Office-CoffeeMail-Task-v0/$i --use_rs

	# rm4 task
	python run.py --alg=qlearning --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql/Office-Long-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql-rs/Office-Long-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm/Office-Long-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm-rs/Office-Long-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm/Office-Long-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-rm4-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm-rs/Office-Long-Task-v0/$i --use_rs

	# Multi-task
	python run.py --alg=qlearning --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql/Office-Multi-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql-rs/Office-Multi-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm/Office-Multi-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm-rs/Office-Multi-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm/Office-Multi-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-multiple-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm-rs/Office-Multi-Task-v0/$i --use_rs
		
	# Rebutal task 1
	python run.py --alg=qlearning --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql/Office-Mod-Coffee-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql-rs/Office-Mod-Coffee-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm/Office-Mod-Coffee-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm-rs/Office-Mod-Coffee-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm/Office-Mod-Coffee-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-up-rm1-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm-rs/Office-Mod-Coffee-Task-v0/$i --use_rs

	# Rebutal task 2
	python run.py --alg=qlearning --env=Office-new-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql/Office-Mod-CoffeeStrict-Task-v0/$i
	python run.py --alg=qlearning --env=Office-new-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/rm-ql-rs/Office-Mod-CoffeeStrict-Task-v0/$i --use_rs
	python run.py --alg=qlearning --env=Office-new-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm/Office-Mod-CoffeeStrict-Task-v0/$i --use_crm
	python run.py --alg=qlearning --env=Office-new-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/crm-rs/Office-Mod-CoffeeStrict-Task-v0/$i --use_crm --use_rs
	python run.py --alg=hrm --env=Office-new-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm/Office-Mod-CoffeeStrict-Task-v0/$i
	python run.py --alg=hrm --env=Office-new-rm5-v0 --num_timesteps=400000 --qinit=0.001 --gamma=0.9 --epsilon 0.5 --print_freq 10000 --init_eval --log_path=../data/logs/hrm-rs/Office-Mod-CoffeeStrict-Task-v0/$i --use_rs
		
	) &> log &
done
