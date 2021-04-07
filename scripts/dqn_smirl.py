"""
Run DQN on grid world.
"""
import sys
try: 
    from comet_ml import Experiment
except: 
    pass

def get_network(network_args, obs_dim, action_dim):
    
    
    if (network_args["type"] == "conv_mixed"):
        from surprise.envs.vizdoom.networks import VizdoomQF
        qf = VizdoomQF(actions=action_dim, **network_args)
        target_qf = VizdoomQF(actions=action_dim, **network_args)
    elif (network_args["type"] == "conv"):
        from surprise.envs.vizdoom.networks import VizdoomFeaturizer
        print ("Using conv")
        qf = VizdoomFeaturizer(dim=action_dim, **network_args)
        target_qf = VizdoomFeaturizer(dim=action_dim, **network_args)
    else:
        from rlkit.torch.networks import Mlp
        qf = Mlp(
            hidden_sizes=[128, 64, 32],
            input_size=obs_dim[0],
            output_size=action_dim,
        )
        target_qf = Mlp(
            hidden_sizes=[128, 64, 32],
            input_size=obs_dim[0],
            output_size=action_dim,
        )
    
    return (qf, target_qf)
    
    
def get_env(variant):
    from launchers.config import BASE_CODE_DIR
    if (variant["env"] == "Tetris"):
        from surprise.envs.tetris.tetris import TetrisEnv
        env = TetrisEnv(render=True, **variant["env_kwargs"])
    elif (variant["env"] == "VizDoom"):
        from surprise.envs.vizdoom.VizdoomWrapper import VizDoomEnv
        env_wargs = variant["env_kwargs"]
        env = VizDoomEnv(config_path=BASE_CODE_DIR + env_wargs["doom_scenario"], 
                         **env_wargs)
    else: 
        import gym
        env = gym.make(variant['env'])
#     else:
#         print("Non supported env type: ", variant["env"])
#         sys.exit()
        
    return env

def add_wrappers(env, variant, device=0, eval=False, network=None):
    from surprise.wrappers.obsresize import ResizeObservationWrapper, RenderingObservationWrapper, SoftResetWrapper, \
        ChannelFirstWrapper, ObsHistoryWrapper
    from surprise.wrappers.VAE_wrapper import VAEWrapper
    obs_dim = env.observation_space.low.shape
    print("obs dim", obs_dim)
    for wrapper in variant["wrappers"]:
        if "soft_reset_wrapper" in wrapper:
            env = SoftResetWrapper(env=env, max_time=500)
        elif "FlattenObservationWrapper" in wrapper:
            from surprise.wrappers.obsresize import FlattenObservationWrapper
            env = FlattenObservationWrapper(env=env, **wrapper["FlattenObservationWrapper"])
        elif "dict_obs_wrapper" in wrapper:
            from surprise.wrappers.obsresize import DictObservationWrapper
            env = DictObservationWrapper(env=env, **wrapper["dict_obs_wrapper"])
        elif "dict_to_obs_wrapper" in wrapper:
            from surprise.wrappers.obsresize import DictToObservationWrapper
            env = DictToObservationWrapper(env=env, **wrapper["dict_to_obs_wrapper"])
        elif ("rendering_observation" in wrapper and 
            (eval == True)
            ):
            env = RenderingObservationWrapper(env=env, **wrapper["rendering_observation"])
        elif "resize_observation_wrapper" in wrapper:
            env = ResizeObservationWrapper(env=env, **wrapper["resize_observation_wrapper"])
            obs_dim = env.observation_space.low.shape
            print("obs dim resize", obs_dim)
        elif "channel_first_observation_wrapper" in wrapper:
            env = ChannelFirstWrapper(env=env, **wrapper["channel_first_observation_wrapper"])
            obs_dim = env.observation_space.low.shape
            print("obs dim channel first", obs_dim)
        elif "ObsHistoryWrapper" in wrapper:
            env = ObsHistoryWrapper(env=env, **wrapper["ObsHistoryWrapper"])
            obs_dim = env.observation_space.low.shape
            print("obs dim history stack", obs_dim)
        elif "vae_wrapper" in wrapper:
            print (wrapper["vae_wrapper"])
            print ("network: ", network)
            env = VAEWrapper(env=env, eval=eval, device=device, network=network, **wrapper["vae_wrapper"])
            network = env.network
        elif "RNDWrapper" in wrapper:
            from surprise.wrappers.RND_wrapper import RNDWrapper
            env = RNDWrapper(env=env, eval=eval, **wrapper["RNDWrapper"], device=device)
            network = env.network
        elif "ICMWrapper" in wrapper:
            from surprise.wrappers.ICM_wrapper import ICMWrapper
            env = ICMWrapper(env=env, eval=eval, **wrapper["ICMWrapper"], device=device)
            network = env.network
        elif "smirl_wrapper" in wrapper:
            env = add_smirl(env=env, variant=wrapper["smirl_wrapper"], device=device)
            
        else:
            if not eval:
                pass
            else:
                print("wrapper not known: ", wrapper)
                sys.exit()
        
    obs_dim = env.observation_space.low.shape
    print("out obs dim", obs_dim)
    return env, network


def add_smirl(env, variant, device=0):
    from surprise.buffers.buffers import BernoulliBuffer, GaussianBufferIncremental, GaussianCircularBuffer
    from surprise.wrappers.base_surprise import BaseSurpriseWrapper
    
    if ("latent_obs_size" in variant):
        obs_size = variant["latent_obs_size"]
    else:
        obs_size = env.observation_space.low.size
        
    if (variant["buffer_type"] == "Bernoulli"):
        buffer = BernoulliBuffer(obs_size)
        env = BaseSurpriseWrapper(env, buffer, time_horizon=100, **variant)
        
    elif (variant["buffer_type"] == "Gaussian"):
#         buffer = GaussianCircularBuffer(obs_size, size=500)
        buffer = GaussianBufferIncremental(obs_size)
        env = BaseSurpriseWrapper(env, buffer, time_horizon=500, **variant)
    else:
        print("Non supported prob distribution type: ", variant["smirl"]["buffer_type"])
        sys.exit()
    
    return env

def experiment(doodad_config, variant):
    from rlkit.core import logger
    from rlkit.launchers.launcher_util import setup_logger
    print ("doodad_config.base_log_dir: ", doodad_config.base_log_dir)
#     setup_logger('ICLR-rebuttal-vizdoom-TakeCover-curiosity', variant=variant)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    setup_logger('wrapped_'+variant['env'], variant=variant, log_dir=doodad_config.base_log_dir+"/smirl/"+variant['exp_name']+"/"+timestamp+"/")
    if (variant["log_comet"]):
        try:
            comet_logger = Experiment(api_key=launchers.config.COMET_API_KEY,
                                         project_name=launchers.config.COMET_PROJECT_NAME, 
                                         workspace=launchers.config.COMET_WORKSPACE)
            logger.set_comet_logger(comet_logger)
            comet_logger.set_name(str(variant['env'])+"_"+str(variant['exp_name']))
            print("variant: ", variant)
            variant['comet_key'] = comet_logger.get_key()
            comet_logger.log_parameters(variant)
            print(comet_logger)
        except Exception as inst:
            print ("Not tracking training via commet.ml")
            print ("Error: ", inst)

    import gym
    from torch import nn as nn
    
    import rlkit.torch.pytorch_util as ptu
    import torch
    from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from rlkit.exploration_strategies.base import \
        PolicyWrappedWithExplorationStrategy
    from rlkit.policies.argmax import ArgmaxDiscretePolicy
    from rlkit.torch.dqn.dqn import DQNTrainer
    from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
    from rlkit.samplers.data_collector import MdpPathCollector
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from surprise.utils.rendering_algorithm import TorchBatchRLRenderAlgorithm
    from surprise.envs.tetris.tetris import TetrisEnv
    from surprise.wrappers.obsresize import ResizeObservationWrapper, RenderingObservationWrapper, SoftResetWrapper
    import pdb
    
    base_env = get_env(variant)
    base_env2 = get_env(variant)
    
    print ("GPU_BUS_Index", variant["GPU_BUS_Index"])
    if torch.cuda.is_available() and doodad_config.use_gpu:
        print ("Using the GPU for learning")
#         ptu.set_gpu_mode(True, gpu_id=doodad_config.gpu_id)
        ptu.set_gpu_mode(True, gpu_id=variant["GPU_BUS_Index"])
    else:
        print ("NOT Using the GPU for learning")
    
#     base_env2 = RenderingObservationWrapper(base_env2)
    expl_env, network = add_wrappers(base_env, variant, device=ptu.device)
    eval_env, _ = add_wrappers(base_env2, variant, device=ptu.device, eval=True, network=network)
    if ("vae_wrapper" in variant["wrappers"]):
        eval_env._network = base_env._network
    
    obs_dim = expl_env.observation_space.low.shape
    print("Final obs dim", obs_dim)
    action_dim = eval_env.action_space.n
    print("Action dimension: ", action_dim)
    qf, target_qf = get_network(variant["network_args"], obs_dim, action_dim)
    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    if "prob_random_action" in variant:
        expl_policy = PolicyWrappedWithExplorationStrategy(
            EpsilonGreedy(expl_env.action_space, prob_random_action=variant["prob_random_action"], 
                          prob_end=variant["prob_end"],
                          steps=variant["steps"]),
            eval_policy,
        )
    else:  
        expl_policy = PolicyWrappedWithExplorationStrategy(
            EpsilonGreedy(expl_env.action_space, prob_random_action=0.8, prob_end=0.05),
            eval_policy,
        )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        render_kwargs=variant['render_kwargs']
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    algorithm = TorchBatchRLRenderAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    from util.simOptions import getOptions
    import sys, os, json, copy
    
#     from doodad.easy_launch.python_function import run_experiment
    
    settings = getOptions(sys.argv)

    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    # experiment(None, variant)
    from util.tuneParams import run_sweep
    from launchers.config import *
    
    sweep_ops={}
    if ( 'tuningConfig' in settings):
        sweep_ops = json.load(open(settings['tuningConfig'], "r"))
        
    run_sweep(experiment, sweep_ops=sweep_ops, variant=settings, repeats=settings['random_seeds'],
          meta_threads=settings['meta_sim_threads'])
