"""Register nuScenes, then run the unmodified official StreetGS training."""
import runpy
from pathlib import Path


def main():
    from lib.datasets import dataset
    from .street_reader import read_nuscenes
    dataset.sceneLoadTypeCallbacks['NuScenes'] = read_nuscenes
    runpy.run_path(str(Path(dataset.__file__).resolve().parents[2] / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
