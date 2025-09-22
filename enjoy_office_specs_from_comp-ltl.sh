#!/bin/bash
#cd reward_machines

python enjoy.py --algo ql --env Office-Mod-Get-Coffee-Task-v0 --sp_dir data/sp_ql/Office-Mod-v0/test/
python enjoy.py --algo ql --env Office-Mod-A-And-D-Task-v0 --sp_dir data/sp_ql/Office-Mod-v0/test/
python enjoy.py --algo ql --env Office-Mod-Not-B-Fn-Task-v0 --sp_dir data/sp_ql/Office-Mod-v0/test/