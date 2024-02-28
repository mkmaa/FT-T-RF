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
from rtdl_revisiting_models import MLP, ResNet, FTTransformer

from adult import Adult
from catboost.datasets import higgs, epsilon

TaskType = Literal["regression", "binclass", "multiclass"]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set random seeds in all libraries.
    # np.random.seed(0)
    delu.random.seed(0)

    # >>> Dataset.

    task_type: TaskType = "regression"
    n_classes = None
    if args.dataset == 'CA':
        dataset = sklearn.datasets.fetch_california_housing() # CA regression
        task_type: TaskType = 'regression'
        n_classes = None
    # elif args.dataset == 'AD':
    #     dataset = Adult(root="datasets", download=True) # AD binclass
    #     task_type: TaskType = 'binclass'
    #     n_classes = 2
    # elif args.dataset == 'HI':
    #     dataset = higgs() # HI binclass
    #     task_type: TaskType = 'binclass'
    #     n_classes = 2
    # elif args.dataset == 'EP':
    #     dataset = epsilon() # EP binclass
    #     task_type: TaskType = 'binclass'
    #     n_classes = 2
    elif args.dataset == 'CO':
        dataset = sklearn.datasets.fetch_covtype() # CO multiclass 7
        task_type: TaskType = 'multiclass'
        n_classes = 7
    elif args.dataset == 'KD':
        dataset = sklearn.datasets.fetch_kddcup99() # KD multiclass 23
        task_type: TaskType = 'multiclass'
        n_classes = 23
    elif args.dataset == 'RC':
        dataset = sklearn.datasets.fetch_rcv1() # RC multiclass 103
        task_type: TaskType = 'multiclass'
        n_classes = 103
    elif args.dataset == 'NE':
        dataset = sklearn.datasets.fetch_20newsgroups_vectorized() # NE multiclass 20
        task_type: TaskType = 'multiclass'
        n_classes = 20
    elif args.dataset == 'LF':
        dataset = sklearn.datasets.fetch_lfw_pairs() # LF binclass
        task_type: TaskType = 'binclass'
        n_classes = 2
    elif args.dataset == 'DI':
        dataset = sklearn.datasets.load_diabetes() # DI regression
        task_type: TaskType = 'regression'
        n_classes = None
    else:
        AssertionError()
    X_cont: np.ndarray = dataset["data"]
    Y: np.ndarray = dataset["target"]
    
    # unique_Y = np.sort(np.unique(Y))
    # Y = np.searchsorted(unique_Y, Y)
    
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
    
    # n_samples = X_cont.shape[0]
    # indices = np.random.permutation(n_samples)
    # X_cont = X_cont[indices]
    # Y = Y[indices]

    # if n_samples > 20000:
    #     X_cont = X_cont[:20000]
    #     Y = Y[:20000]

    # >>> Continuous features.
    X_cont: np.ndarray = X_cont.astype(np.float32)
    n_cont_features = X_cont.shape[1]

    # >>> Categorical features.
    # NOTE: the above datasets do not have categorical features, but,
    # for the demonstration purposes, it is possible to generate them.
    cat_cardinalities = [
        # NOTE: uncomment the two lines below to add two categorical features.
        # 4,  # Allowed values: [0, 1, 2, 3].
        # 7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
    ]
    X_cat = (
        np.column_stack(
            [np.random.randint(0, c, (len(X_cont),)) for c in cat_cardinalities]
        )
        if cat_cardinalities
        else None
    )

    # >>> Labels.
    # Regression labels must be represented by float32.
    if task_type == "regression":
        Y = Y.astype(np.float32)
    else:
        assert n_classes is not None
        Y = Y.astype(np.int64)
        assert set(Y.tolist()) == set(
            range(n_classes)
        ), "Classification labels must form the range [0, 1, ..., n_classes - 1]"

    # >>> Split the dataset.
    all_idx = np.arange(len(Y))
    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.8
    )
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        trainval_idx, train_size=0.8
    )
    data_numpy = {
        "train": {"x_cont": X_cont[train_idx], "y": Y[train_idx]},
        "val": {"x_cont": X_cont[val_idx], "y": Y[val_idx]},
        "test": {"x_cont": X_cont[test_idx], "y": Y[test_idx]},
    }
    if X_cat is not None:
        data_numpy["train"]["x_cat"] = X_cat[train_idx]
        data_numpy["val"]["x_cat"] = X_cat[val_idx]
        data_numpy["test"]["x_cat"] = X_cat[test_idx]
        
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
        n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
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

    # print('n_feature: ', n_cont_features, 'cat_card: ', cat_cardinalities, 'd_out: ', d_out)

    # # NOTE: uncomment to train MLP
    # model = MLP(
    #     d_in=n_cont_features + sum(cat_cardinalities),
    #     d_out=d_out,
    #     n_blocks=2,
    #     d_block=384,
    #     dropout=0.1,
    # ).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # # NOTE: uncomment to train ResNet
    # model = ResNet(
    #     d_in=n_cont_features + sum(cat_cardinalities),
    #     d_out=d_out,
    #     n_blocks=2,
    #     d_block=192,
    #     d_hidden=None,
    #     d_hidden_multiplier=2.0,
    #     dropout1=0.3,
    #     dropout2=0.0,
    # ).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    model = FTTransformer(
        FTTargs=args,
        n_cont_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        **FTTransformer.get_default_kwargs(n_blocks=6),
    ).to(device)
    optimizer = model.make_default_optimizer()

    def apply_model(batch: Dict[str, Tensor]) -> Tensor:
        if isinstance(model, (MLP, ResNet)):
            x_cat_ohe = (
                [
                    F.one_hot(column, cardinality)
                    for column, cardinality in zip(batch["x_cat"].T, cat_cardinalities)
                ]
                if "x_cat" in batch
                else []
            )
            return model(torch.column_stack([batch["x_cont"]] + x_cat_ohe)).squeeze(-1)

        elif isinstance(model, FTTransformer):
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
    epoch_size = math.ceil(len(train_idx) / batch_size)
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
    parser.add_argument('--dataset', type=str, choices=['CA', 'CO', 'KD', 'RC', 'NE', 'LF', 'DI'], help='dataset')
    parser.add_argument('--rf', type=str, choices=['True', 'False'], help='random feature')
    parser.add_argument('--bias', type=str, choices=['True', 'False'], help='bias, only in uniform init')
    parser.add_argument('--activation', type=str, default='ReGLU', choices=['ReGLU', 'ReLU'], help='ReGLU or ReLU')
    parser.add_argument('--init', type=str, choices=['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'], help='Xavier or Kaiming, uniform or normal')
    parser.add_argument('--fan', type=str, choices=['fan_in', 'fan_out'], help='fan_in or fan_out, only in Kaiming init')
    
    args = parser.parse_args()
    main(args)