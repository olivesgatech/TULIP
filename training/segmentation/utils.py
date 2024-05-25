import os
import pandas as pd
import numpy as np
from training.segmentation.segmentationtracker import SegmentationTracker


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def gen_folder(name: str):
    folder = os.path.expanduser(name)
    if not os.path.exists(folder):
        os.makedirs(folder)


class TestOutput:
    def __init__(self, out_df: pd.DataFrame, out_tracker: SegmentationTracker, prediction: np.ndarray):
        self.statdf = out_df
        self.tracker = out_tracker
        self.prediction = prediction
