# ruff: noqa: E402
import math
import warnings
from typing import Dict, Literal

warnings.simplefilter("ignore")
import delu  # Deep Learning Utilities: https://github.com/Yura52/delu
import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
import argparse

warnings.resetwarnings()
from rtdl_revisiting_models import FTTransformer

from read_data import read_dataset

TaskType = Literal["regression", "binclass", "multiclass"]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set random seeds in all libraries.
    delu.random.seed(0)

    # >>> Dataset.
    D, N, C, Y, y_info = read_dataset(args)

    task_type: TaskType = D.info['task_type']
    n_classes = None
    
    # if args.dataset == 'CA':
    #     dataset = sklearn.datasets.fetch_california_housing() # CA regression
    #     task_type: TaskType = 'regression'
    #     n_classes = None
    # elif args.dataset == 'CO':
    #     dataset = sklearn.datasets.fetch_covtype() # CO multiclass 7
    #     task_type: TaskType = 'multiclass'
    #     n_classes = 7
    # elif args.dataset == 'KD':
    #     dataset = sklearn.datasets.fetch_kddcup99() # KD multiclass 23
    #     task_type: TaskType = 'multiclass'
    #     n_classes = 23
    # elif args.dataset == 'RC':
    #     dataset = sklearn.datasets.fetch_rcv1() # RC multiclass 103
    #     task_type: TaskType = 'multiclass'
    #     n_classes = 103
    # elif args.dataset == 'NE':
    #     dataset = sklearn.datasets.fetch_20newsgroups_vectorized() # NE multiclass 20
    #     task_type: TaskType = 'multiclass'
    #     n_classes = 20
    # elif args.dataset == 'LF':
    #     dataset = sklearn.datasets.fetch_lfw_pairs() # LF binclass
    #     task_type: TaskType = 'binclass'
    #     n_classes = 2
    # elif args.dataset == 'DI':
    #     dataset = sklearn.datasets.load_diabetes() # DI regression
    #     task_type: TaskType = 'regression'
    #     n_classes = None
    # else:
    #     AssertionError()
    # X_cont: np.ndarray = dataset["data"]
    # Y: np.ndarray = dataset["target"]
    
    # NOTE: uncomment to solve a classification task.
    # if (task_type != 'regression'):
    #     assert n_classes != None and n_classes >= 2
    #     task_type: TaskType = 'binclass' if n_classes == 2 else 'multiclass'
    #     X_cont, Y = sklearn.datasets.make_classification(
    #         n_samples=20000,
    #         n_features=8,
    #         n_classes=n_classes,
    #         n_informative=6,
    #         n_redundant=2,
    #     )
    
    # if args.dataset == 'CO':
    #     unique_Y = np.sort(np.unique(Y))
    #     Y = np.searchsorted(unique_Y, Y)
        
    #     n_samples = X_cont.shape[0]
    #     indices = np.random.permutation(n_samples)
    #     X_cont = X_cont[indices]
    #     Y = Y[indices]
    #     if n_samples > 20000:
    #         X_cont = X_cont[:20000]
    #         Y = Y[:20000]

    # >>> Continuous features.
    # X_cont: np.ndarray = X_cont.astype(np.float32)
    # n_cont_features = X_cont.shape[1]

    # >>> Categorical features.
    # NOTE: the above datasets do not have categorical features, but,
    # for the demonstration purposes, it is possible to generate them.
    # cat_cardinalities = [
    #     # NOTE: uncomment the two lines below to add two categorical features.
    #     # 4,  # Allowed values: [0, 1, 2, 3].
    #     # 7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
    # ]
    # X_cat = (
    #     np.column_stack(
    #         [np.random.randint(0, c, (len(X_cont),)) for c in cat_cardinalities]
    #     )
    #     if cat_cardinalities
    #     else None
    # )

    # >>> Labels.
    # Regression labels must be represented by float32.
    if task_type == "regression":
        Y['train'] = Y['train'].astype(np.float32)
        Y['val'] = Y['val'].astype(np.float32)
        Y['test'] = Y['test'].astype(np.float32)
    else:
        assert n_classes is not None
        Y = Y.astype(np.int64)
        assert set(Y.tolist()) == set(
            range(n_classes)
        ), "Classification labels must form the range [0, 1, ..., n_classes - 1]"

    data_numpy = {
        "train": {"y": Y['train']},
        "val": {"y": Y['val']},
        "test": {"y": Y['test']},
    }
    if N is not None:
        data_numpy["train"]["x_cont"] = N['train'].astype(np.float32)
        data_numpy["val"]["x_cont"] = N['val'].astype(np.float32)
        data_numpy["test"]["x_cont"] = N['test'].astype(np.float32)
    if C is not None:
        data_numpy["train"]["x_cat"] = C['train'].astype(np.int64)
        data_numpy["val"]["x_cat"] = C['val'].astype(np.int64)
        data_numpy["test"]["x_cat"] = C['test'].astype(np.int64)
        
    train_idx = (len(N['train']) if N is not None else 0) + (len(C['train']) if C is not None else 0)
    # val_idx = len(N['val']) + len(C['val'])
    # test_idx = len(N['test']) + len(C['test'])
        
    # >>> Feature preprocessing.
    # NOTE
    # The choice between preprocessing strategies depends on a task and a model.

    # (A) Simple preprocessing strategy.
    # preprocessing = sklearn.preprocessing.StandardScaler().fit(
    #     data_numpy['train']['x_cont']
    # )

    # (B) Fancy preprocessing strategy.
    # The noise is added to improve the output of QuantileTransformer in some cases.
    X_cont_train_numpy = data_numpy["train"]["x_cont"]
    noise = (
        np.random.default_rng(0)
        .normal(0.0, 1e-5, X_cont_train_numpy.shape)
        .astype(X_cont_train_numpy.dtype)
    )
    preprocessing = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(train_idx // 30, 1000), 10),
        output_distribution="normal",
        subsample=10**9,
    ).fit(X_cont_train_numpy + noise)
    del X_cont_train_numpy

    for part in data_numpy:
        data_numpy[part]["x_cont"] = preprocessing.transform(data_numpy[part]["x_cont"])

    # >>> Label preprocessing.
    if task_type == "regression":
        Y_mean = data_numpy["train"]["y"].mean().item()
        Y_std = data_numpy["train"]["y"].std().item()
        for part in data_numpy:
            data_numpy[part]["y"] = (data_numpy[part]["y"] - Y_mean) / Y_std

    # >>> Convert data to tensors.
    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
        for part in data_numpy
    }

    if task_type != "multiclass":
        # Required by F.binary_cross_entropy_with_logits
        for part in data:
            data[part]["y"] = data[part]["y"].float()

    # The output size.
    d_out = n_classes if task_type == "multiclass" else 1

    model = FTTransformer(
        FTTargs=args,
        n_cont_features=D.info['n_num_features'],
        cat_cardinalities=D.info['cat_cardinalities'],
        d_out=d_out,
        **FTTransformer.get_default_kwargs(n_blocks=args.n_blocks),
    ).to(device)
    optimizer = model.make_default_optimizer()

    def apply_model(batch: Dict[str, Tensor]) -> Tensor:
        if isinstance(model, FTTransformer):
            return model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)
        else:
            raise RuntimeError(f"Unknown model type: {type(model)}")


    loss_fn = (
        F.binary_cross_entropy_with_logits
        if task_type == "binclass"
        else F.cross_entropy
        if task_type == "multiclass"
        else F.mse_loss
    )


    @torch.no_grad()
    def evaluate(part: str) -> float:
        model.eval()

        eval_batch_size = 8096
        y_pred = (
            torch.cat(
                [
                    apply_model(batch)
                    for batch in delu.iter_batches(data[part], eval_batch_size)
                ]
            )
            .cpu()
            .numpy()
        )
        y_true = data[part]["y"].cpu().numpy()

        if task_type == "binclass":
            y_pred = np.round(scipy.special.expit(y_pred))
            score = sklearn.metrics.accuracy_score(y_true, y_pred)
        elif task_type == "multiclass":
            y_pred = y_pred.argmax(1)
            score = sklearn.metrics.accuracy_score(y_true, y_pred)
        else:
            assert task_type == "regression"
            score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5 * Y_std)
        return score  # The higher -- the better.

    print(f'Test score before training: {evaluate("test"):.4f}')

    # For demonstration purposes (fast training and bad performance),
    # one can set smaller values:
    # n_epochs = 20
    # patience = 2
    n_epochs = 1000000
    patience = 16

    batch_size = 256
    epoch_size = math.ceil(train_idx / batch_size)
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(patience, mode="max")
    best = {
        "val": -math.inf,
        "test": -math.inf,
        "epoch": -1,
    }

    print(f"Device: {device.type.upper()}")
    print("-" * 88 + "\n")
    timer.run()
    for epoch in range(n_epochs):
        for batch in tqdm(
            delu.iter_batches(data["train"], batch_size, shuffle=True),
            desc=f"Epoch {epoch}",
            total=epoch_size,
        ):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model(batch), batch["y"])
            loss.backward()
            optimizer.step()

        val_score = evaluate("val")
        test_score = evaluate("test")
        print(f"(val) {val_score:.4f} (test) {test_score:.4f} [time] {timer}")

        early_stopping.update(val_score)
        if epoch >= 80 and early_stopping.should_stop():
            break

        if val_score > best["val"]:
            print("🌸 New best epoch! 🌸")
            best = {"val": val_score, "test": test_score, "epoch": epoch}
        print()

    print("\n\nResult:")
    print(best)
    with open("log.txt", "a") as f:
        print(args, file=f)
        print(best, file=f)
        print('', file=f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input args')
    parser.add_argument('--dataset', type=str, choices=['CA', 'CO', 'KD', 'RC', 'NE', 'LF', 'DI', 'r1', 'r2'], help='dataset')
    parser.add_argument('--rf_in', type=str, choices=['True', 'False'], help='inner random feature, replace feature tokenizer')
    parser.add_argument('--rf_out', type=str, choices=['True', 'False'], help='outler random feature, work with feature tokenizer')
    parser.add_argument('--bias', type=str, choices=['True', 'False'], help='bias, only in uniform init')
    parser.add_argument('--activation', type=str, default='ReGLU', choices=['ReGLU', 'ReLU'], help='ReGLU or ReLU')
    parser.add_argument('--init', type=str, choices=['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'], help='Xavier or Kaiming, uniform or normal')
    parser.add_argument('--fan', type=str, choices=['fan_in', 'fan_out'], help='fan_in or fan_out, only in Kaiming init')
    parser.add_argument('--n_blocks', type=int, default=3, choices=[1, 2, 3, 4, 5, 6], help='n_blocks of token embedding in FT-T')
    
    parser.add_argument('--normalization', type=str, default='standard', help='Normalization method')
    parser.add_argument('--num_nan_policy', type=str, default='mean', help='How to deal with missing values in numerical features')
    parser.add_argument('--cat_nan_policy', type=str, default='most_frequent', help='How to deal with missing values in categorical features')
    parser.add_argument('--cat_policy', type=str, default='onehot', help='How to deal with categorical features')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    if args.init == 'kaiming_uniform' or 'kaiming_normal':
        if args.fan == None:
            raise AssertionError('please input fan mode')
    
    main(args)