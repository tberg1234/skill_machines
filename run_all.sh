#!/bin/bash
for i in `seq 0 9`; 
do
	(
	# Learn primitives
	python sm_ql.py --env Office-v0 --total_steps 100000 --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sp_ql/Office-v0/$i/
	python sm_ql.py --env Office-Mod-v0 --total_steps 100000 --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sp_ql/Office-Mod-v0/$i/
		
	# Office-World tasks
	python sm_ql.py --env Office-Coffee-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/Office-Coffee-Task-v0/$i/
	python sm_ql.py --env Office-Coffee-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-Coffee-Task-v0/$i/
	python sm_ql.py --env Office-Coffee-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-Coffee-Task-v0/$i/
	python ql.py --env Office-Coffee-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-Coffee-Task-v0/$i/
	
	python sm_ql.py --env Office-CoffeeMail-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/Office-CoffeeMail-Task-v0/$i/
	python sm_ql.py --env Office-CoffeeMail-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-CoffeeMail-Task-v0/$i/
	python sm_ql.py --env Office-CoffeeMail-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-CoffeeMail-Task-v0/$i/
	python ql.py --env Office-CoffeeMail-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-CoffeeMail-Task-v0/$i/
	
	python sm_ql.py --env Office-Long-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/Office-Long-Task-v0/$i/
	python sm_ql.py --env Office-Long-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-Long-Task-v0/$i/
	python sm_ql.py --env Office-Long-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-Long-Task-v0/$i/
	python ql.py --env Office-Long-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-Long-Task-v0/$i/
	
	python sm_ql.py --env Office-Multi-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/Office-Multi-Task-v0/$i/
	python sm_ql.py --env Office-Multi-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-Multi-Task-v0/$i/
	python sm_ql.py --env Office-Multi-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-Multi-Task-v0/$i/
	python ql.py --env Office-Multi-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-Multi-Task-v0/$i/
		
	python sm_ql.py --env Office-Mod-Coffee-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/Office-Mod-Coffee-Task-v0/$i/
	python sm_ql.py --env Office-Mod-Coffee-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-Mod-Coffee-Task-v0/$i/
	python sm_ql.py --env Office-Mod-Coffee-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-Mod-Coffee-Task-v0/$i/
	python ql.py --env Office-Mod-Coffee-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-Mod-Coffee-Task-v0/$i/
	
	python sm_ql.py --env Office-Mod-CoffeeStrict-Task-v0 --total_steps 400000 --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/Office-Mod-CoffeeStrict-Task-v0/$i/
	python sm_ql.py --env Office-Mod-CoffeeStrict-Task-v0 --total_steps 400000 --zeroshot --load --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/zeroshot/Office-Mod-CoffeeStrict-Task-v0/$i/
	python sm_ql.py --env Office-Mod-CoffeeStrict-Task-v0 --total_steps 400000 --fewshot --load --sp_dir data/sp_ql/Office-Mod-v0/$i/ --log_dir data/logs/sm_ql/fewshot/Office-Mod-CoffeeStrict-Task-v0/$i/
	python ql.py --env Office-Mod-CoffeeStrict-Task-v0 --total_steps 400000 --log_dir data/logs/ql/Office-Mod-CoffeeStrict-Task-v0/$i/

        ) &> log_office &
done

for i in `seq 0 4`;
do
        (	
	# Learn primitives
	python sm_sb3.py --env MovingTargets-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sp_dqn/$i/

	# Moving-Targets tasks
        python sm_sb3.py --env MovingTargets-Task-1-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-Task-1-v0/$i/ --load
        python sm_sb3.py --env MovingTargets-Task-1-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-Task-1-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-Task-1-v0/$i/
        python sb3.py --env MovingTargets-Task-1-v0 --algo dqn --skill_dir data/dqn/MovingTargets-Task-1-v0/$i/ --log_dir data/logs/dqn/MovingTargets-Task-1-v0/$i/

        python sm_sb3.py --env MovingTargets-Task-4-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-Task-4-v0/$i/ --load
        python sm_sb3.py --env MovingTargets-Task-4-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-Task-4-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-Task-4-v0/$i/
        python sb3.py --env MovingTargets-Task-4-v0 --algo dqn --skill_dir data/dqn/MovingTargets-Task-4-v0/$i/ --log_dir data/logs/dqn/MovingTargets-Task-4-v0/$i/

        python sm_sb3.py --env MovingTargets-Multi-Task-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-Multi-Task-v0/$i/ --load
        python sm_sb3.py --env MovingTargets-Multi-Task-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-Multi-Task-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-Multi-Task-v0/$i/
        python sb3.py --env MovingTargets-Multi-Task-v0 --algo dqn --skill_dir data/dqn/MovingTargets-Multi-Task-v0/$i/ --log_dir data/logs/dqn/MovingTargets-Multi-Task-v0/$i/

        python sm_sb3.py --env MovingTargets-PartiallyOrdered-Task-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-PartiallyOrdered-Task-v0/$i/ --load
        python sm_sb3.py --env MovingTargets-PartiallyOrdered-Task-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-PartiallyOrdered-Task-v0/$i/ --log_dir data/logs/sm_dqn/MovingTargets-PartiallyOrdered-Task-v0/$i/
        python sb3.py --env MovingTargets-PartiallyOrdered-Task-v0 --algo dqn --skill_dir data/dqn/MovingTargets-PartiallyOrdered-Task-v0/$i/ --log_dir data/logs/dqn/MovingTargets-PartiallyOrdered-Task-v0/$i/

        ) &> log_moving
done &

for i in `seq 0 4`;
do
	(
	# Learn primitives
        python sm_sb3.py --env Safety-v0 --algo td3 --sp_dir data/sp_td3/Safety-v0/$i/ --log_dir data/logs/sp_td3/Safety-v0/$i/
		
	# Safety-gym tasks
        python sm_sb3.py --env Safety-Task-1-v0 --algo td3 --sp_dir data/sp_td3/Safety-v0/$i/ --log_dir data/logs/sm_td3/Safety-Task-1-v0/$i/ --load
        python sm_sb3.py --env Safety-Task-1-v0 --algo td3 --sp_dir data/sp_td3/Safety-Task-1-v0/$i/ --log_dir data/logs/sm_td3/Safety-Task-1-v0/$i/
        python sb3.py --env Safety-Task-1-v0 --algo td3 --skill_dir data/td3/Safety-Task-1-v0/$i/ --log_dir data/logs/td3/Safety-Task-1-v0/$i/

        python sm_sb3.py --env Safety-Task-4-v0 --algo td3 --sp_dir data/sp_td3/Safety-v0/$i/ --log_dir data/logs/sm_td3/Safety-Task-4-v0/$i/ --load
        python sm_sb3.py --env Safety-Task-4-v0 --algo td3 --sp_dir data/sp_td3/Safety-Task-4-v0/$i/ --log_dir data/logs/sm_td3/Safety-Task-4-v0/$i/
        python sb3.py --env Safety-Task-4-v0 --algo td3 --skill_dir data/td3/Safety-Task-4-v0/$i/ --log_dir data/logs/td3/Safety-Task-4-v0/$i/

        python sm_sb3.py --env Safety-Multi-Task-v0 --algo td3 --sp_dir data/sp_td3/Safety-v0/$i/ --log_dir data/logs/sm_td3/Safety-Multi-Task-v0/$i/ --load
        python sm_sb3.py --env Safety-Multi-Task-v0 --algo td3 --sp_dir data/sp_td3/Safety-Multi-Task-v0/$i/ --log_dir data/logs/sm_td3/Safety-Multi-Task-v0/$i/
        python sb3.py --env Safety-Multi-Task-v0 --algo td3 --skill_dir data/td3/Safety-Multi-Task-v0/$i/ --log_dir data/logs/td3/Safety-Multi-Task-v0/$i/

        python sm_sb3.py --env Safety-PartiallyOrdered-Task-v0 --algo td3 --sp_dir data/sp_td3/Safety-v0/$i/ --log_dir data/logs/sm_td3/Safety-PartiallyOrdered-Task-v0/$i/ --load
        python sm_sb3.py --env Safety-PartiallyOrdered-Task-v0 --algo td3 --sp_dir data/sp_td3/Safety-PartiallyOrdered-Task-v0/$i/ --log_dir data/logs/sm_td3/Safety-PartiallyOrdered-Task-v0/$i/
        python sb3.py --env Safety-PartiallyOrdered-Task-v0 --algo td3 --skill_dir data/td3/Safety-PartiallyOrdered-Task-v0/$i/ --log_dir data/logs/td3/Safety-PartiallyOrdered-Task-v0/$i/    

	) &> log_safety
done &
