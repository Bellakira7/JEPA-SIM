"""Convert an official, trusted UniAD motion-anchor pickle into the port's NPZ format."""
import argparse
from pathlib import Path
import pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, type=Path, help='Official checkpoint-matched anchor pickle')
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Only load anchor files obtained from the official model provider.
    with args.input.open('rb') as stream:
        item = pickle.load(stream)
    anchors = item['anchors_all'] if isinstance(item, dict) else item
    anchors = np.asarray(anchors)
    if anchors.ndim != 4 or anchors.shape[-2:] != (12, 2) or not np.isfinite(anchors).all():
        raise ValueError(f'Expected finite [groups, modes, 12, 2] anchors; got {anchors.shape}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez(stream, anchors_all=anchors)
    print(args.output)


if __name__ == '__main__':
    main()
