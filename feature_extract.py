import argparse
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader

from dataset import MyDataset
from extractor import Extractor


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset_dir', type=str, default='/media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/front')
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--save_dir', type=str, default='/media/moon/T7 Shield/multiview_results')

    options = args.parse_args()

    loader = DataLoader(MyDataset(Path(options.dataset_dir)),
                        batch_size = options.batch_size,
                        num_workers = 0)
    
    extractor = Extractor(loader)

    extractor.feature_extract()

    np.save(options.save_dir, extractor.get_matrix())


if __name__ == '__main__':
    main()