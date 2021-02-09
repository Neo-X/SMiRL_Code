# Bayesian Surprise

Repo for environments, gym wrappers, and scripts for the SMiRL project.


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