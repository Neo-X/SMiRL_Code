

# from doodad.easy_sweep.hyper_sweep import run_sweep_parallel, run_sweep_doodad
# import doodad

class Sweeper(object):
    def __init__(self, hyper_config, repeat=1, include_name=False):
        self.hyper_config = hyper_config
        self.hyper_config['random_seed'] = list(range(repeat))
        self.include_name=include_name

    def __iter__(self):
        import itertools
        count = 0
        for config in itertools.product(*[val for val in self.hyper_config.values()]):
            kwargs = {key:config[i] for i, key in enumerate(self.hyper_config.keys())}
            if self.include_name:
                timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                kwargs['exp_name'] = "%s_%d" % (timestamp, count)
            count += 1
            yield kwargs
            
def run_exp(variant):
    from doodad.easy_launch.python_function import run_experiment
#     from launchers.config import *
    method, variant = variant
    
    print ("variant: ", variant)
    run_experiment(
        method,
        exp_name=variant["exp_name"],
#         mode='local_docker',
        mode=variant["doodad_run_mode"],
#         mode='local',
#         mode='ec2',
        ssh_host=variant["ssh_host"],
        use_gpu=variant["training_processor_type"] == "gpu",
        variant=variant,
        region='us-east-2',
    )

def run_sweep(experiment, sweep_ops, variant, repeats=2, meta_threads=4):
    import multiprocessing
        ### Need to convert parameters into iterables
    for key in variant.keys():
        variant[key] = [variant[key]] 
        
    ## Update sweep parameters
    for key in sweep_ops.keys():
        variant[key] = sweep_ops[key]
        
    print("variant: ", variant)
    sweeper = Sweeper(variant, repeat=repeats)
    
    print ("meta_threads: ", meta_threads)
#     sys.exit()
    pool = multiprocessing.Pool(meta_threads)
    exp_args = []
    for config in sweeper:
        exp_args.append((experiment, config))
#     random.shuffle(exp_args)
    pool.map(run_exp, exp_args)
    

def example_run_method(**kwargs):
    import time
    time.sleep(1.0)
    print(kwargs)
    
    
    
if __name__ == '__main__':    

    
    import os, sys
    from util.simOptions import getOptions
    import json
    settings = getOptions(sys.argv)
    
    sweep_op = json.load(open(settings['tuningConfig'], "r"))
    CoMPS_PATH = os.environ['CoMPS_PATH']
    MULTIWORLD_PATH = os.environ['MULTIWORLD_PATH']
    
    from comet_ml import Experiment, ExistingExperiment
    from doodad.easy_launch.python_function import run_experiment
    meta_comet_logger = meta_comet_key = rl_comet_logger = rl_comet_key = None
    
    
    from functional_scripts.gmps_train import experiment
    general_settings = settings["general"]
    from datetime import datetime
    now = datetime.now()
    print("path_to_gmps: ", CoMPS_PATH)
    test_dir = os.path.join(CoMPS_PATH, general_settings["test_dir"], str(now), "")
    BASE_EXPERT_LOC = os.path.join(CoMPS_PATH, general_settings["BASE_EXPERT_LOC"], "")


    meta_variant = settings["meta"]
    meta_variant["expertDataLoc"] = BASE_EXPERT_LOC
#     meta_variant["log_dir"] = test_dir
    meta_variant['comet_key'] = meta_comet_key
    meta_variant["CoMPS_PATH"] = CoMPS_PATH
    meta_variant["MULTIWORLD_PATH"] = MULTIWORLD_PATH
    meta_variant['outer_iteration'] = 0
    meta_variant['off_policy'] = False
    meta_variant['test_dir'] = general_settings['test_dir']
    meta_variant['log_comet'] = settings["log_comet"]
    meta_variant['exp_name'] = settings['exp_name']

    print("Starting experiment")
    print ("meta_variant: ", meta_variant)
    
#     run_sweep_parallel(run_method=experiment, 
#                      params=meta_variant, 
#                      repeat=2,
#                      num_cpu=4
#                      )
    
    run_sweep(experiment, sweep_op, meta_variant)
    
#     experiment(None, meta_variant)
#     run_experiment(
#         experiment,
#         exp_name=settings["exp_name"],
#         mode='local_docker',
# #         mode='local',
# #         mode='ec2',
#         variant=meta_variant,
#         region='us-east-2',
#     )

#     mounts = [
#     '/home/gberseth/playground/CoMPS',
#     '/home/gberseth/playground/R_multiworld',
#     ]
#     
#     mount_ = doodad.mount.MountLocal(
#         local_dir='/home/gberseth/playground/CoMPS', # The name of this directory on disk
#         mount_point='/home/gberseth/playground/CoMPS', # The name of this directory as visible to the script
#         pythonpath=True, # Whether to add this directory to the pythonpath
#         output=False, # Whether this directory is an empty directory for storing outputs.
#         filter_ext=['.pyc', '.log', '.git', '.mp4'], # File extensions to not include
#         filter_dir=[] # Directories to ignore
#     )
#     
#     mode_ = doodad.mode.EC2AutoconfigDocker(
#                 region='us-west-1',
#                 s3_bucket='comps-test',
#                 image_id='gberseth/glen-image:comps',
#                 aws_key_name=os.environ['AWS_ACCESS_KEY'],
#                 iam_profile=os.environ['AWS_ACCESS_SECRET']
#                 )
#     run_sweep_doodad(run_method=experiment, 
#                      params=meta_variant, 
#                      repeat=1,
#                      run_mode=mode_,
#                      mounts=mount_
#                      )
    
    