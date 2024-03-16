import os
import dataclasses as dc
import typing as ty
import json
import argparse
import numpy as np
import sklearn.preprocessing as skp
from copy import deepcopy
from pathlib import Path
from sklearn.impute import SimpleImputer
from scipy.stats import mode

import torch

BINCLASS = 'binclass'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'

ArrayDict = ty.Dict[str, np.ndarray]

def load_json(path):
    return json.loads(Path(path).read_text())

def normalize(
    X: ArrayDict, normalization: str, seed: int
) -> ArrayDict:
    X_train = X['train'].copy()
    if normalization == 'standard':
        normalizer = skp.StandardScaler()
    elif normalization == 'minmax':
        normalizer = skp.MinMaxScaler()
    elif normalization == 'maxabs':
        normalizer = skp.MaxAbsScaler()
    elif normalization == 'normal':
        normalizer = skp.Normalizer()
    elif normalization == 'robust':
        normalizer = skp.RobustScaler()
    else:
        # TODO 在此处实现数据预处理方法
        raise ValueError('No such option')
    normalizer.fit(X_train)
    ret: ArrayDict = {k: normalizer.transform(v) for k, v in X.items()}
    return ret

@dc.dataclass
class Dataset:
    N: ty.Optional[ArrayDict]
    C: ty.Optional[ArrayDict]
    y: ArrayDict
    info: ty.Dict[str, ty.Any]
    folder: ty.Optional[Path]

    @classmethod
    def from_dir(cls, dir_: ty.Union[Path, str]) -> 'Dataset':
        dir_ = Path(dir_)

        def load(item) -> ArrayDict:
            return {
                x: ty.cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy', allow_pickle=True))  # type: ignore[code]
                for x in ['train', 'val', 'test']
            }

        return Dataset(
            load('N') if dir_.joinpath('N_train.npy').exists() else None,
            load('C') if dir_.joinpath('C_train.npy').exists() else None,
            load('y'),
            load_json(dir_ / 'info.json'),
            dir_,
        )

    @property
    def is_binclass(self) -> bool:
        return self.info['task_type'] == BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.info['task_type'] == MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.info['task_type'] == REGRESSION

    @property
    def n_num_features(self) -> int:
        return self.info['n_num_features']

    @property
    def n_cat_features(self) -> int:
        return self.info['n_cat_features']

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features
    
    @property
    def dataset_scale(self) -> int:
        return self.info['train_samples'] + self.info['val_samples'] + self.info['test_samples']

    def build_X(
        self,
        *,
        normalization: ty.Optional[str],
        num_nan_policy: str,
        cat_nan_policy: str,
        cat_policy: str,
        seed: int,
    ) -> ty.Union[ArrayDict, ty.Tuple[ArrayDict, ArrayDict]]:

        if self.N:
            N = deepcopy(self.N)

            num_nan_masks = {k: np.isnan(v) for k, v in N.items()}
            if any(x.any() for x in num_nan_masks.values()): 
                if num_nan_policy == 'mean':
                    num_new_values = np.nanmean(self.N['train'], axis=0)
                elif num_nan_policy == 'media':
                    num_new_values = np.nanmedian(self.N['train'], axis=0)
                elif num_nan_policy == 'mode':
                    num_new_values = mode(self.N['train'], axis=0).mode
                else:
                    # TODO 在此处实现对连续值缺失处理的方法
                    raise ValueError('No such option')
                for k, v in N.items():
                    num_nan_indices = np.where(num_nan_masks[k])
                    v[num_nan_indices] = np.take(num_new_values, num_nan_indices[1])
            if normalization != 'none':
                N = normalize(N, normalization, seed)

        else:
            N = None

        if not self.C:
            assert N is not None
            return N, None

        C = deepcopy(self.C)

        imputer = SimpleImputer(strategy=cat_nan_policy) # default='most_frequent'
        imputer.fit(C['train'])
        if imputer:
            C = {k: imputer.transform(v) for k, v in C.items()}

        if cat_policy == 'onehot':
            onehot = skp.OneHotEncoder(
                handle_unknown='ignore', sparse_output=False, dtype='float32'  # type: ignore[code]
            )
            onehot.fit(C['train'])
            C = {k: onehot.transform(v) for k, v in C.items()}
            return N, C
        else:
            # TODO 在这里实现对category features的编码
            raise ValueError('No such option')

    def build_y(
        self
    ) -> ty.Tuple[ArrayDict, ty.Optional[ty.Dict[str, ty.Any]]]:
        y = deepcopy(self.y)        
        if self.is_regression:
            # regression
            mean, std = self.y['train'].mean(), self.y['train'].std()
            y = {k: (v - mean) / std for k, v in y.items()}
            info = {'policy': 'mean_std', 'mean': mean, 'std': std}
            return y, info
        else:
            # classification
            # print('Before: ', y)
            encoder = skp.LabelEncoder()
            encoder.fit(y['train'])
            y['train'] = encoder.transform(y['train'])
            y['test'] = encoder.transform(y['test'])
            y['val'] = encoder.transform(y['val'])
            # print('After : ', y)
            return y, {'policy': 'none'}

def read_dataset(args):
    # print(args)
    D = Dataset.from_dir(os.path.join('data', args.dataset))
    N, C = D.build_X(  
        normalization = args.normalization,
        num_nan_policy = args.num_nan_policy,
        cat_nan_policy = args.cat_nan_policy,
        cat_policy = args.cat_policy,
        seed=args.seed
    )
    Y, y_info = D.build_y()
    return D, N, C, Y, y_info

def read_XTab_dataset_train(dataset):
    args = argparse.Namespace(
        dataset='../../Data-Preprocess-Tabular-Data/data_full/'+dataset, 
        normalization='standard', 
        num_nan_policy='mean', 
        cat_nan_policy='most_frequent', 
        cat_policy='onehot',
        seed=0
    )
    D, N, C, Y, y_info = read_dataset(args=args)
    N_tot = np.vstack((N['train'], N['val'], N['test'])) if N is not None else None
    C_tot = np.vstack((C['train'], C['val'], C['test'])) if C is not None else None
    if N_tot is None and C_tot is None:
        raise AssertionError('X_tot is None')
    elif N_tot is None:
        X_tot = C_tot
    elif C_tot is None:
        X_tot = N_tot
    else:
        X_tot = np.hstack((N_tot, C_tot))
    Y_tot = np.hstack((Y['train'], Y['val'], Y['test']))
    # X_tot.astype(np.float32)
    # Y_tot.astype(np.float32)
    # print('X_tot', X_tot)
    # print('Y_tot', Y_tot)
    if X_tot.shape[0] != Y_tot.shape[0]:
        raise AssertionError('the shape of X_tot and Y_tot are different')
    n_feature = X_tot.shape[1]
    X_tot = torch.from_numpy(X_tot).float()
    Y_tot = torch.from_numpy(Y_tot).float()
    return D.info['task_type'], X_tot[:10000], Y_tot[:10000], n_feature

def read_XTab_dataset_test(dataset):
    args = argparse.Namespace(
        dataset='../../Data-Preprocess-Tabular-Data/data_full/'+dataset, 
        normalization='standard', 
        num_nan_policy='mean', 
        cat_nan_policy='most_frequent', 
        cat_policy='onehot',
        seed=0
    )
    D, N, C, Y, y_info = read_dataset(args=args)
    N_train = N['train'] if N is not None else None
    N_test = N['val'] if N is not None else None
    # N_test = np.vstack((N['val'], N['test'])) if N is not None else None
    C_train = C['train'] if C is not None else None
    C_test = C['val'] if C is not None else None
    # C_test = np.vstack((C['val'], C['test'])) if C is not None else None
    if N is None and C is None:
        raise AssertionError('X_tot is None')
    elif N is None:
        X_train = C_train
        X_test = C_test
    elif C is None:
        X_train = N_train
        X_test = N_test
    else:
        X_train = np.hstack((N_train, C_train))
        X_test = np.hstack((N_test, C_test))
    Y_train = Y['train']
    Y_test = Y['val']
    # Y_test = np.hstack((Y['val'], Y['test']))
    if X_train.shape[0] != Y_train.shape[0] or X_test.shape[0] != Y_test.shape[0]:
        raise AssertionError('the shape of X_tot and Y_tot are different')
    n_feature = X_train.shape[1]
    n_samples = X_train.shape[0]
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    if D.info['task_type'] == 'multiclass':
        Y_train = torch.from_numpy(Y_train).long()
        Y_test = torch.from_numpy(Y_test).long()
        d_out = D.info['num_class']
    else:
        Y_train = torch.from_numpy(Y_train).float()
        Y_test = torch.from_numpy(Y_test).float()
        d_out = 1
    return D.info['task_type'], X_train, X_test, Y_train, Y_test, n_feature, d_out