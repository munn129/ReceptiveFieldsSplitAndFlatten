import argparse
import numpy as np
import faiss

from pathlib import Path
from tqdm import tqdm

feature_dir = 'extracted feature directory'
