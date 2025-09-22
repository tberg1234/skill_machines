#!/bin/bash
for i in `seq 0 5`;
do
        (	
	# Learn primitives
	python sm_sb3.py --env MovingTargets-v0 --algo dqn --sp_dir data/sp_dqn/MovingTargets-v0/$i/ --log_dir data/logs/sp_dqn/$i/

    ) &> log_moving
done

python enjoy.py --algo dqn --env StaticTargets-BlueSquareTask-v0 --sp_dir data/sp_dqn/MovingTargets-v0/test/
python enjoy.py --algo dqn --env StaticTargets-PurpleCircleTask-v0 --sp_dir data/sp_dqn/MovingTargets-v0/test/
python enjoy.py --algo dqn --env StaticTargets-BlueNotSquareTask-v0 --sp_dir data/sp_dqn/MovingTargets-v0/test/
python enjoy.py --algo dqn --env StaticTargets-BlueSquareAndPurpleCircleTask-v0 --sp_dir data/sp_dqn/MovingTargets-v0/test/
&