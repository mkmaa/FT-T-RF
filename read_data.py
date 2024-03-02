import os
import dataclasses as dc
import typing as ty
import json
import numpy as np
import sklearn.preprocessing as skp
from copy import deepcopy
from pathlib import Path
from sklearn.impute import SimpleImputer
from scipy.stats import mode

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