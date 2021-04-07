# Bayesian Surprise

Repo for environments, gym wrappers, and scripts for the SMiRL project.


## Requirements:

doodad: https://github.com/Neo-X/doodad/

rlkit: https://github.com/Neo-X/rlkit/tree/surprise

### Build Instruction

```
conda create --name smirl python=3.5 pip
conda activate smirl
pip install -r requirements.txt
```

Then you will need to updated the paths in the `launchers.config.py`. 
You need to update `BASE_CODE_DIR` to the location you have saved SMiRL_Code.
Also update `LOCAL_LOG_DIR` to the location you would like the logging data to be saved on your computer.
You can look at the [doodad](https://github.com/Neo-X/doodad/) for more details on this configuration.

## Commands:

###Run Vizdoom SMiRL experiments

python3 scripts/dqn_smirl.py --config=configs/VizDoom_TakeCover_Small.json --exp_name=vizdoom_small_test --run_mode=ssh --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

 python3 scripts/dqn_smirl.py --config=configs/VizDoom_DefendTheLine_Small.json --exp_name=vizdoom_DTL_small_smirl --run_mode=ssh  --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

 python3 scripts/dqn_smirl.py --config=configs/VizDoom_DefendTheLine_Small_Bonus.json --exp_name=vizdoom_DTL_small_smirl_bonus --run_mode=ssh --ssh_host=newton1 --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

### Run Atari Experiments

python3 scripts/dqn_smirl.py --config=configs/Carnival_Small_SMiRL.json --exp_name=Atari_Carnival__small_smirl --run_mode=ssh  --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

python3 scripts/dqn_smirl.py --config=configs/Carnival_Small_SMiRL_Bonus.json --exp_name=Atari_Carnival_small_smirl_bonus --run_mode=ssh --ssh_host=newton1 --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

python3 scripts/dqn_smirl.py --config=configs/IceHockey_Small_SMiRL.json --exp_name=Atari_IceHockey_small_smirl --run_mode=ssh  --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json

python3 scripts/dqn_smirl.py --config=configs/RiverRaid_Small_SMiRL.json --exp_name=Atari_RiverRaid_small_smirl --run_mode=ssh --ssh_host=newton1 --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json
