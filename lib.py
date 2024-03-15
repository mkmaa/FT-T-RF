import numpy as np
import scipy
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA

from rtdl_revisiting_models import _CLSEmbedding


@torch.no_grad()
def RandomFeaturePreprocess(
        X_train: torch.Tensor, 
        X_test: torch.Tensor, 
        d_embedding: int, n_dims: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    X = torch.cat((X_train, X_test), dim=0)
    weight = torch.empty(X.shape[1], d_embedding)
    nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
    X = X @ weight
    X = nn.ReLU()(X)
    pca = PCA(n_components=n_dims)
    X = torch.from_numpy(pca.fit_transform(X.cpu().numpy())).to(X.device)
    return torch.split(X, [n_samples_train, n_samples_test], dim=0)


@torch.no_grad()
def Evaluate(task_type, y_pred, y_true):
    if task_type == "binclass":
        y_pred = np.round(scipy.special.expit(y_pred))
        score = accuracy_score(y_true, y_pred)
    elif task_type == "multiclass":
        y_pred = y_pred.argmax(1)
        score = accuracy_score(y_true, y_pred)
    else:
        assert task_type == "regression"
        score = -(mean_squared_error(y_true, y_pred))
    return score


class FeatureTokenizer(nn.Module): # FT-T: CLS + linear embedding
    def __init__(self, n_features: int, n_dims: int) -> None:
        super().__init__()
        self.cls = _CLSEmbedding(n_dims)
        self.weight = torch.nn.Parameter(torch.empty(n_features, n_dims))
        self.bias = torch.nn.Parameter(torch.empty(n_features, n_dims))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )
        x_cls = self.cls(x.shape[:-1])
        x = x[..., None] * self.weight
        x = x + self.bias[None]
        # print(f'{x_cls.shape=}, {x.shape=}')
        return torch.cat([x_cls, x], dim=1)