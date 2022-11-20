"""
Copy-pasted from https://github.com/universome/fvd-comparison/blob/master/our_fvd.py & 
https://github.com/universome/fvd-comparison/blob/master/compare_metrics.py
which had a reference to
https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
from typing import Tuple
import scipy
import numpy as np
import torch
import io
#import pickle

'''
def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]

    return mu, sigma

@torch.no_grad()
def compute_fvd(videos_fake: np.ndarray, videos_real: np.ndarray, device: str='cuda') -> float:
    #detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

    # Load model from io.BytesIO object
    with open('models/i3d_torchscript.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
        #torch.jit.load(buffer)
        detector = torch.jit.load(buffer).eval().to(device)
    
    #with open_url(detector_url, verbose=False) as f:
    #    detector = torch.jit.load(f).eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()

    return compute_fvd(feats_fake, feats_real)
'''

@torch.no_grad()
def load_detector(device='cuda'):
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

    # Load model from io.BytesIO object
    with open('models/i3d_torchscript.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
        #torch.jit.load(buffer)
        detector = torch.jit.load(buffer).eval().to(device)
    
    return detector

def get_FeatureStats():
    stats = FeatureStats(capture_mean_cov=True)
    return stats

def update_stats_for_sequence(detector, featurestats, videos, device='cuda'):
    videos = videos.permute(0, 4, 1, 2, 3).to(device).to(torch.float)
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    #print('videos', videos.shape)
    feats = detector(videos, **detector_kwargs)#.cpu().numpy()

    featurestats.append_torch(feats)
    #print('updating stats')

def compute_fvd(feats_stats_fake, feat_stats_real) -> float:
    #print('feats_stats_fake', feats_stats_fake.shape,feats_stats_real.shape)
    mu_gen, sigma_gen = feats_stats_fake.get_mean_cov() #compute_stats(feats_fake)
    mu_real, sigma_real = feat_stats_real.get_mean_cov() #compute_stats(feats_real)
    print('mu', mu_gen, mu_real)
    print('sigma', sigma_gen, sigma_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    print('m', m, 's', s)
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj


    