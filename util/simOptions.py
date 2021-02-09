from optparse import OptionParser
import sys

def getOptions(_args=None):
    parser = OptionParser()

    parser.add_option("--load_saved_model", "--continue_training",
              action="store", dest="load_saved_model", default=None,
              # type=int,
              choices=['true', 'false', 'best', 'last', None],
              metavar="STRING", 
              help="Should the system load a pretrained model [true|false|network_and_scales]")
    
    parser.add_option("--train_forward_dynamics",
              action="store", dest="train_forward_dynamics", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="Whether or not to train a forward dynamics model as well [true|false|None]")
    
    parser.add_option("--folder_instance_name",
              action="store", dest="folder_instance_name",
              default=None,
              metavar="STRING", 
              help="Folder instance name. This is some suffix to help prevent folder name clashes when running new simulations")
    
    parser.add_option("--extra_opts",
              action="store", dest="opts",
              default=None,
              metavar="STRING", 
              help="Extra options that are given to the simulation when executed on the command line.")
    
    
    parser.add_option("--experiment_logging",
              action="store", dest="experiment_logging",
              default=None,
              metavar="STRING", 
              help="Details to define what kind of logging should be don for this simulation.")
    
    
    parser.add_option("--train_actor",
              action="store", dest="train_actor", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="Whether or not to train the policy. Used for debugging")
    
    parser.add_option("--train_critic",
              action="store", dest="train_critic", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="Whether or not to train the value function. Used for debugging")
    
    parser.add_option("--skip_rollouts",
              action="store", dest="skip_rollouts", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="A flag to turn off data collections phases (rollouts) after initial bootstrapping.")
    
    parser.add_option("--email_log_data_periodically",
              action="store", dest="email_log_data_periodically", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="Whether or not to periodically email the simulation learning data so far [true|false|None]")
    
    parser.add_option("--test_movie_rendering",
              action="store", dest="test_movie_rendering", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="Test movie rendering [true|false|None]")
    
    parser.add_option("--model_perform_batch_training", "--perform_batch_training",
              action="store", dest="model_perform_batch_training", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="Whether or not to perform training in epochs [true|false|None]")
    
    parser.add_option("--Use_Multi_GPU_Simulation", "--Multi_GPU",
              action="store", dest="Use_Multi_GPU_Simulation", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="Whether or not to perform training in epochs [true|false|None]")
    
    parser.add_option("--rnn_updates",
              action="store", dest="rnn_updates", default=None,
              # type='choice',
              # choices=['true', 'false', None],
              metavar="INTEGER", 
              help="Override the number of FD RNN updates.")
    
    parser.add_option("--GPU_BUS_Index",
              action="store", dest="GPU_BUS_Index", default=0,
              # type='choice',
              # choices=['true', 'false', None],
              metavar="INTEGER", 
              help="Index of GPU to perform training on.")
    
    parser.add_option("--force_sim_net_to_cpu",
              action="store", dest="force_sim_net_to_cpu", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="Even if performing training on the GPU the networks for the simulations will be put on the CPU.")
    
    parser.add_option("--keep_seperate_fd_exp_buffer",
              action="store", dest="keep_seperate_fd_exp_buffer", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="If to use a separate exp mem for the fd model, often to store more data for on-policy methods.")
    
    parser.add_option("--on_policy",
              action="store", dest="on_policy", default=None,
              type='choice',
              choices=['true', 'false', 'fast', None],
              metavar="STRING", 
              help="Whether or not to perform training in epochs [true|false|None]")
    
    parser.add_option("--reset_on_fall",
              action="store", dest="reset_on_fall", default=None,
              type='choice',
              choices=['true', 'false', None],
              metavar="STRING", 
              help="Whether or not the controller should be reset to a new epoch when a fall (fallen into some kind of non-recoverable state) has occurred")
    
    parser.add_option("--visualize_expected_value",
              action="store", dest="visualize_expected_value", default=None,
              type='choice',
              choices=['production', 'staging', 'true', 'false', None],
              metavar="STRING", 
              help="Should the system load a pretrained model [true|false|network_and_scales]")
    
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="print status messages to stdout")
    
    parser.add_option("--dataDir", "--datastoreDirectory", "--dataPath", "--data_folder",
                  action="store", dest="data_folder", default=None,
                  metavar="Directory", help="Specify the directory that files will be stored")
    
    parser.add_option("-p", "--processes", "--availableProcesses", "--n_parallel",
                      "--p",
              action="store", dest="n_parallel", default=None,
              type=int,
              metavar="INTEGER", help="The number of processes the SteerStats script can use")

    parser.add_option("--simulation_timeout",
              action="store", dest="simulation_timeout", default=None,
              type=int,
              metavar="INTEGER", help="The number of seconds queues will wait for simulations")
    
    parser.add_option("--email_logging_time",
              action="store", dest="email_logging_time", default=None,
              type=int,
              metavar="INTEGER", help="The number of seconds between policy progress emails are sent")
    
    parser.add_option("--additional_on_policy_training_updates",
              action="store", dest="additional_on_policy_training_updates", default=None,
              type=int,
              metavar="INTEGER", help="Additional on-policy training updates")
    
    parser.add_option("--batch_size",
              action="store", dest="batch_size", default=None,
              type=int,
              metavar="INTEGER", help="Batch size used for policy learning")
    
    parser.add_option("--fd_updates_per_actor_update",
              action="store", dest="fd_updates_per_actor_update", default=None,
              type=int,
              metavar="INTEGER", help="The number of fd network to be done per actor update.")
    
    parser.add_option("--pretrain_critic",
              action="store", dest="pretrain_critic", default=None,
              type=int,
              metavar="INTEGER", help="Run some initial training steps to pretrain the critic before starting policy training.")
    
    parser.add_option("--pretrain_fd",
              action="store", dest="pretrain_fd", default=None,
              type=int,
              metavar="INTEGER", help="Run some initial training steps to pretrain the fd before starting policy training.")
    
    parser.add_option("--meta_sim_samples", "--mp", "--random_seeds",
              action="store", dest="random_seeds", default=1,
              type=int,
              metavar="INTEGER",
              help="Number of simulation samples to compute to use for Meta simulation, running a number of simulations and taking the average of them")
    
    parser.add_option("--num_on_policy_rollouts", "--rollouts",
              action="store", dest="num_on_policy_rollouts", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of on policy rollouts to perform per epoch.")
    
    parser.add_option("--meta_sim_threads", "--mt",
              action="store", dest="meta_sim_threads", default=1,
              type=int,
              metavar="INTEGER", 
              help="Number of threads to use for Meta simulation, running a number of simulations and taking the average of them")
    
    parser.add_option("--tuning_threads", "--tt",
              action="store", dest="tuning_threads", default=1,
              type=int,
              metavar="INTEGER", 
              help="Number of threads to use for hyper parameter tuning")
    

    parser.add_option("--num_param_samples", "--nps",
              action="store", dest="num_param_samples", default=1,
              type=int,
              metavar="INTEGER", 
              help="Number of samples to evaluation between the given bounds")
        
    parser.add_option("--bootstrap_samples",
              action="store", dest="bootstrap_samples", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of initial actions to sample before calculating input/output scaling and starting to train.")
    
    parser.add_option("--experience_length",
              action="store", dest="experience_length", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of samples that will fit into the exp buffer (circular queue).")
    
    parser.add_option("--eval_epochs",
              action="store", dest="eval_epochs", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of epoch/episode to evaluate the policy over")
    
    parser.add_option("--epochs",
              action="store", dest="epochs", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of epochs to perform per round")
    
    parser.add_option("--max_path_length",
              action="store", dest="max_path_length", default=None,
              type=int,
              metavar="INTEGER", 
              help="That max number of action that can be take before the end of an episode/epoch")
    
    parser.add_option("--print_level",
              action="store", dest="print_level", default=None,
#              type=string,
              metavar="STRING", 
              help="Controls the level of information that is printed to the terminal")
    
    parser.add_option("--n_itr",
              action="store", dest="n_itr", default=None,
              type=int,
              metavar="INTEGER", 
              help="Controls the number of simulation rounds")
    
    parser.add_option("--plotting_update_freq_num_rounds",
              action="store", dest="plotting_update_freq_num_rounds", default=None,
              type=int,
              metavar="INTEGER", 
              help="Controls the number of simulation rounds to perform before evaluating and re-plotting the policy performance")
    
    parser.add_option("--saving_update_freq_num_rounds",
              action="store", dest="saving_update_freq_num_rounds", default=None,
              type=int,
              metavar="INTEGER", 
              help="Controllers the number of simulation rounds to perform before saving the policy")
    
    parser.add_option("--sim_config_file", "--env",
              action="store", dest="env", default=None,
              # type=int,
              metavar="STRING", 
              help="Path to the file the contains the settings for the simulation")
    
    parser.add_option("--save_video_to_file",
              action="store", dest="save_video_to_file", default=None,
              # type=int,
              metavar="STRING", 
              help="Path to the file that a video will be create after the end of the training")
    
    parser.add_option("--frameSize", 
          action="store", dest="frameSize", default=None,
          metavar="IntegerxInteger", help="The pixel width and height, example 640x480")
    
    parser.add_option("--visualize_learning", "--plot", 
          action="store", dest="visualize_learning", default=None,
          type='choice',
          choices=['true', 'false', None],
          metavar="STRING", 
          help="Whether or not to draw/render the simulation")
    
    parser.add_option("--learning_backend", "--backend", 
          action="store", dest="learning_backend", default=None,
          # type='choice',
          # choices=['true', 'false', None],
          metavar="STRING", 
          help="Which backend to use for keras [theano|tensorflow]")
    
    parser.add_option("--save_trainData", 
          action="store", dest="save_trainData", default=None,
          type='choice',
          choices=['true', 'false', None],
          metavar="STRING", 
          help="Whether or not to save plots of the training results during learning")
    
    parser.add_option("--save_experience_memory", "--save_experience", 
          action="store", dest="save_experience_memory", default=None,
          type='choice',
          choices=['true', 'false', 'continual', 'all', None],
          metavar="STRING", 
          help="Whether or not to save the experience memory after performing initial bootstrapping.")
    
    parser.add_option("--shouldRender", "--render",
          action="store", dest="shouldRender", default=None,
          type='choice',
          choices=['true', 'false', None, 'yes'],
          metavar="STRING", 
          help="TO specify if an openGL window should be created")

    parser.add_option("--config", 
           action="store", metavar="STRING", dest="configFile", default=None,
          help="""The json config file that many of the config settings can be parsed from""")
    
    parser.add_option("--tuningConfig", 
           action="store", metavar="STRING", dest="tuningConfig", default=None,
          help="""The json config file that many of the config settings can be parsed from json for tuning parameters""")
    
    parser.add_option("--randomSeed", 
           action="store", dest="randomSeed", metavar="INTEGER", default=None,
           help="""randomSeed that will be used for random scenario generation.""")

    parser.add_option("--checkpoint_vid_rounds",
           action="store", dest="checkpoint_vid_rounds", metavar="INTEGER", default=None,
           help="""Controls the number of simulation rounds to perform before saving a video generated from the policy.""")
    
    parser.add_option("--log_comet",
          action="store", dest="log_comet", default='false',
          type='choice',
          choices=['true', 'false'],
          metavar="STRING", 
          help="TO specify if learning should try to log data to a comet.ml experiment")

    parser.add_option("--name", "--exp_name",
                      action="store", metavar="STRING", dest="exp_name", default=None,
                      help="""the name showing up on comet""")
    
    parser.add_option("--training_processor_type",
                      action="store", metavar="STRING", dest="training_processor_type", default="cpu",
                      type='choice',
                      choices=['cpu', 'gpu'],
                      help="""the name showing up on comet""")
    
    parser.add_option("--run_mode",
              action="store", dest="doodad_run_mode", default='local',
              type='choice',
              choices=['local', 'local_docker', 'ssh', 'ec2', 'local_singularity', 'slurm_singularity'],
              metavar="STRING", 
              help="Method for running code for doodad")
    
    parser.add_option("--ssh_host",
                  action="store", metavar="STRING", dest="ssh_host", default="default",
                  help="""the name showing up on comet""")
    
    if _args is None:
        (options, args) = parser.parse_args()
    else:
        (options, args) = parser.parse_args(_args)

    ### convert to vars
    options = vars(options)

    ### Convert to dictionary
    options = processOptions(options)
    return options
# print getOptions()

def processOptions(options):
    ### Convert options into a dictionary
    import json 
    
    file = open(options['configFile'])
    settings = json.load(file)
    file.close()
    
    for option in options:
        if ( not (options[option] is None) ):
            print ("Updateing option: ", option, " = ", options[option])
            # settings[option] = json.loads(options[option])
            settings[option] = options[option]
            try:
                settings[option] = json.loads(settings[option])
            except Exception as e:
                pass # dataTar.close()
            if ( options[option] == 'true'):
                settings[option] = True
            elif ( options[option] == 'false'):
                settings[option] = False
    
    return settings

