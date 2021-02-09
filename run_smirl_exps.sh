#!/bin/bash

### Run all the smirl experiments

### Tertris
SEEDS=6
# python3 scripts/dqn_smirl.py --config=configs/tetris_SMiRL.json --exp_name=tetris_smirl --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true
# python3 scripts/dqn_smirl.py --config=configs/tetris_RND.json --exp_name=tetris_rnd --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS
# python3 scripts/dqn_smirl.py --config=configs/tetris_ICM.json --exp_name=tetris_icm --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS
# python3 scripts/dqn_smirl.py --config=configs/tetris.json --exp_name=tetris --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS
# python3 scripts/dqn_smirl.py --config=configs/tetris_SMiRL_VAE_append.json --exp_name=tetris_smirl_vae_append --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS
python3 scripts/dqn_smirl.py --config=configs/tetris_ICM_SMiRL.json --exp_name=tetris_icm_smirl --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS
# python3 scripts/dqn_smirl.py --config=configs/tetris_rows_cleared.json --exp_name=tetris_rows_cleared --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS
#### Bonus rewards
# python3 scripts/dqn_smirl.py --config=configs/tetris/tetris_ICM_bonus.json --exp_name=tetris_ICM_bonus --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS
# python3 scripts/dqn_smirl.py --config=configs/tetris/tetris_RND_bonus.json --exp_name=tetris_RND_bonus --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS
# python3 scripts/dqn_smirl.py --config=configs/tetris/tetris_SMiRL_bonus.json --exp_name=tetris_SMiRL_bonus --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS
# python3 scripts/dqn_smirl.py --config=configs/tetris/tetris_SMiRL_plus_ICM_bonus.json --exp_name=tetris_SMiRL_plus_ICM_bonus --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS
# python3 scripts/dqn_smirl.py --config=configs/tetris/tetris_SMiRL_VAE_append_bonus.json --exp_name=tetris_SMiRL_VAE_bonus --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true --meta_sim_threads=$SEEDS




###VizDoom
# PYTHONPATH=/home/gberseth/playground/doodad_vitchry/ python3 scripts/dqn_smirl.py --config=configs/VizDoom_TakeCover_Small.json --exp_name=VizDoom_noGod_smirl --run_mode=ssh --ssh_host=newton1 --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json
# PYTHONPATH=/home/gberseth/playground/doodad_vitchry/ python3 scripts/dqn_smirl.py --config=configs/VizDoom_TakeCover_Small_Bonus.json --exp_name=VizDoom_noGod_smirl_Bonus --run_mode=ssh --ssh_host=newton5 --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json
# PYTHONPATH=/global/scratch/gberseth/playground/BayesianSurpriseCode/ python3 scripts/dqn_smirl.py --config=configs/VizDoom_TakeCover_RND.json --exp_name=vizdoom_RND --tuningConfig=configs/tuning/VizDoom_Envs.json --meta_sim_samples=6 --meta_sim_threads=1 --run_mode=slurm_singularity --training_processor_type=gpu --log_comet=true
# PYTHONPATH=/global/scratch/gberseth/playground/BayesianSurpriseCode/ python3 scripts/dqn_smirl.py --config=configs/VizDoom_TakeCover_SMiRL_plus_ICM.json --exp_name=vizdoom_SMiRL_plus_ICM --tuningConfig=configs/tuning/VizDoom_Envs.json --meta_sim_samples=6 --meta_sim_threads=1 --run_mode=slurm_singularity --training_processor_type=gpu --log_comet=true
# python3 scripts/dqn_smirl.py --config=configs/VizDoom_TakeCover_SMiRL_plus_ICM.json --exp_name=SMiRL_plus_ICM
#Bonus
# python3 scripts/dqn_smirl.py --config=configs/VizDoom/VizDoom_DefendTheLine_SMiRL_VAE_Bonus.json --exp_name=vizdoom_SMiRL_VAE_bonus --random_seeds=$SEEDS --run_mode=ec2 --log_comet=true
# PYTHONPATH=/global/scratch/gberseth/playground/BayesianSurpriseCode/ python3 scripts/dqn_smirl.py --config=configs/VizDoom/VizDoom_TakeCover_ICM_bonus.json --exp_name=vizdoom_ICM_bonus --tuningConfig=configs/tuning/VizDoom_Envs.json --meta_sim_samples=6 --meta_sim_threads=1 --run_mode=slurm_singularity --training_processor_type=gpu --log_comet=true
# PYTHONPATH=/global/scratch/gberseth/playground/BayesianSurpriseCode/ python3 scripts/dqn_smirl.py --config=configs/VizDoom/VizDoom_TakeCover_RND_bonus.json --exp_name=vizdoom_RND_bonus --tuningConfig=configs/tuning/VizDoom_Envs.json --meta_sim_samples=6 --meta_sim_threads=1 --run_mode=slurm_singularity --training_processor_type=gpu --log_comet=true

### Atarti Games
# PYTHONPATH=/home/gberseth/playground/doodad_vitchry/ python3 scripts/dqn_smirl.py --config=configs/IceHockey_Small_SMiRL.json --exp_name=Atari_IceHockey_small_smirl --run_mode=ssh --ssh_host=newton5 --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json
# PYTHONPATH=/home/gberseth/playground/doodad_vitchry/ python3 scripts/dqn_smirl.py --config=configs/RiverRaid_Small_SMiRL.json --exp_name=Atari_RiverRaid_small_smirl --run_mode=ssh --ssh_host=newton1 --random_seeds=1 --meta_sim_threads=4 --log_comet=true --training_processor_type=gpu --tuningConfig=configs/GPU_indexes.json
# PYTHONPATH=/global/scratch/gberseth/playground/BayesianSurpriseCode/ python3 scripts/dqn_smirl.py --config=configs/Atari_RND.json --exp_name=Atari_RND --tuningConfig=configs/tuning/Atari_Envs.json --meta_sim_samples=3 --meta_sim_threads=1 --run_mode=slurm_singularity --training_processor_type=gpu --log_comet=true
# PYTHONPATH=/global/scratch/gberseth/playground/BayesianSurpriseCode/ python3 scripts/dqn_smirl.py --config=configs/Atari_SMiRL.json --exp_name=Atari_SMiRL --tuningConfig=configs/tuning/Atari_Envs.json --meta_sim_samples=3 --meta_sim_threads=1 --run_mode=slurm_singularity --training_processor_type=gpu --log_comet=true
# PYTHONPATH=/global/scratch/gberseth/playground/BayesianSurpriseCode/ python3 scripts/dqn_smirl.py --config=configs/Atari_ICM.json --exp_name=Atari_ICM --tuningConfig=configs/tuning/Atari_Envs.json --meta_sim_samples=3 --meta_sim_threads=1 --run_mode=slurm_singularity --training_processor_type=gpu --log_comet=true
