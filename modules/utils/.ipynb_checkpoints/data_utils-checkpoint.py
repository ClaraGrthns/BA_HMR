
import numpy as np


def get_relevant_keypoints(keypoints, threshold=0.3):
    return [(x,y) for (x,y,relevance) in (keypoints).transpose(1,0) if relevance>threshold]