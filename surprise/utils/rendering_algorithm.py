

from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
# from torch.utils.tensorboard import SummaryWriter
# from railrl.core import logger
from collections import OrderedDict
# from rlkit.core.logging import append_log
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
from rlkit.core import logger
from util.utils import current_mem_usage

def display_gif(images, logdir, fps=10, max_outputs=8, counter=0):
    ### image format (episodes, img_width, img_height, colour_channels)
    import moviepy.editor as mpy
    import numpy as np
    print ("images shape: ", images.shape)
    images = images[:max_outputs]
    images = np.concatenate(images, axis=-2)
    clip = mpy.ImageSequenceClip(list(images), fps=fps)
#     clip.write_gif(logdir+str(counter)+".gif", fps=fps)
#     clip.write_videofile(logdir+str(counter)+".webm", fps=fps)
#     clip.write_videofile(logdir+".mp4", fps=fps)
    clip.write_videofile(logdir+str(counter)+".mp4", fps=fps)
    cl = logger.get_comet_logger()
    if (cl is not None):
#             cl.set_step(step=epoch)
#         cl.log_image(image_data=logdir+".mp4", overwrite=True, image_format="mp4")
        cl.log_image(image_data=logdir+str(counter)+".mp4", overwrite=True, image_format="mp4")

class TorchBatchRLRenderAlgorithm(TorchBatchRLAlgorithm):
    
    
    def _train(self):
        
#         pdb.set_trace()

        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            cl = logger.get_comet_logger()
            if (cl is not None):
                cl.set_step(step=epoch)
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)
            
            if ((epoch % 25) == 0):
                print("Rendering video")
                self.render_video("eval_video_", counter=epoch)
            self._end_epoch(epoch)
        
#         algo_log = OrderedDict()
#         append_log(algo_log, self.expl_data_collector.get_diagnostics(), prefix='exploration/')
        
#         for k,v in algo_log.items():
#             if type(v) is not OrderedDict:
#                 self.writer.add_scalar(k, v, self.epoch)
#             else:
#                 # For putting two plots on same graph where v is dict of scalars
#                 self.writer.add_scalars(k, v, self.epoch)
#             logger.record_tabular('current_mem_usage', current_mem_usage())
            
        return out
        
    def render_video(self, tag, counter):
        import numpy as np
        import pdb
#         log.debug("{}".format("render_video_and_add_to_tensorboard"))
        
        path = self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True
        )
        
        if ("vae_reconstruction" in path[0]['env_infos'][0]):
            video = np.array([ [y['vae_reconstruction'] for y in x['env_infos']] for x in  path])
            display_gif(images=video, logdir=logger.get_snapshot_dir()+"/"+tag+"_reconstruction" , fps=15, counter=counter)
            
        video = np.array([ [y['rendering'] for y in x['env_infos']] for x in  path])
        display_gif(images=video, logdir=logger.get_snapshot_dir()+"/"+tag , fps=15, counter=counter)
            
        
