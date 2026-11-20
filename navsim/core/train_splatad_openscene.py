"""Launch the native SplatAD trainer with the OpenScene ADDataParser."""
import argparse
from copy import deepcopy
from pathlib import Path
from openscene_splat_parser import OpenSceneParserConfig
from nerfstudio.configs.method_configs import method_configs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--steps',type=int,default=30001)
    a=ap.parse_args()
    config=deepcopy(method_configs['splatad'])
    config.output_dir=a.out
    config.experiment_name='openscene'
    config.timestamp='run'
    config.vis='tensorboard'
    config.max_num_iterations=a.steps
    config.steps_per_eval_all_images=100000000
    config.steps_per_save=a.steps
    config.pipeline.model.max_steps=a.steps
    config.pipeline.model.implementation='torch'
    config.pipeline.datamanager.dataparser=OpenSceneParserConfig(data=a.data)
    config.pipeline.datamanager.camera_res_scale_factor=.5
    config.save_config()
    trainer=config.setup(local_rank=0,world_size=1)
    trainer.setup()
    trainer.train()


if __name__=='__main__':main()
