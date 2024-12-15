import argparse
import numpy as np
import faiss

from pathlib import Path
from tqdm import tqdm

index_feature_dir = 'extracted index feature directory'
query_feature_dir = 'extracted query feature directory'

image_dir_prefix = ''
index_dir = f'{image_dir_prefix}/ INDEX DIR'
query_dir = f'{image_dir_prefix}/ QUERY DIR'

def image_namelist_generator(image_dir):
    image_namelist = sorted(list(Path(image_dir).glob('*.png')))
    return [str(i) for i in image_namelist]

def main():

    result = []

    index_image_namelist = image_namelist_generator(index_dir)
    query_image_namelist = image_namelist_generator(query_dir)

    index_feature = np.load(index_feature_dir).astype('float32')
    query_feature = np.load(query_feature_dir).astype('float32')

    pool_size = query_feature.shape[1]

    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(index_feature)

    _, predictions = faiss.search(query_feature, min(len(index_feature), 1))

    for idx, val in enumerate(predictions):
        result.append(f'{query_image_namelist[idx]} {index_image_namelist[val[0]]}')

    with open('result.txt', 'a') as file:
        for line in result:
            file.write(line + '\n')


if __name__ == '__main__':
    main()