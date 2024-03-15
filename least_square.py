import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from tqdm.std import tqdm
import delu
import math
from datetime import datetime

from read_data import read_XTab_dataset_test
from rtdl_revisiting_models import FTTransformer, FTTransformerBackbone
from main import RandomFeaturePreprocess, Evaluate, FeatureTokenizer

def LeastSquare(X_train, Y_train, X_test, alpha=0.0): # Linear Regression: use tensor to maintain gradients
    X_train = torch.cat((X_train, torch.ones(X_train.shape[0], 1)), dim=1)
    X_test = torch.cat((X_test, torch.ones(X_test.shape[0], 1)), dim=1)
    I = torch.eye(X_train.shape[1])
    X_pinv = torch.inverse(X_train.t() @ X_train + alpha * I) @ X_train.t()
    theta = X_pinv @ Y_train
    Y_pred = X_test @ theta
    return Y_pred


class Model_leastsq(nn.Module):
    def __init__(self, num_datasets, n_features_list, args):
        super(Model_leastsq, self).__init__()
        if args.mode != 'test_lsq' or num_datasets != 1:
            raise AssertionError('testing multiple datasets')
        self.num_datasets = num_datasets
        self.ft = FeatureTokenizer(n_features=args.n_pca, n_dims=args.n_dims)
        kwargs = FTTransformer.get_default_kwargs(n_blocks=args.n_blocks)
        del kwargs['_is_default']
        kwargs['d_out'] = None
        self.backbone = FTTransformerBackbone(**kwargs)

    def forward(self, X):
        X = self.ft(X)
        X = self.backbone(X)
        return X


def test_leastsq(args):
    task_type, X_train, X_test, Y_train, Y_test, n = read_XTab_dataset_test('__public__/' + args.dataset)
    print(f'Started training. Training size: {X_train.shape[0]}, testing size: {X_test.shape[0]}, feature number: {X_train.shape[1]}.')
    X_train, X_test = RandomFeaturePreprocess(X_train, X_test, d_embedding=args.d_embedding, n_dims=args.n_pca)
    print(f'Started training. After RF. Training size: {X_train.shape[0]}, testing size: {X_test.shape[0]}, feature number: {X_train.shape[1]}.')
    
    if task_type != 'regression' and task_type != 'binclass':
        raise AssertionError('not regression or binclass')
    
    if task_type == 'binclass':     # for binclass least square
        Y_train = Y_train * 2 - 1
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'device: {device}')
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    Y_train = Y_train.to(device)
    
    test_model = Model_leastsq(num_datasets=1, n_features_list=[n], args=args).to(device)
    test_model.backbone.load_state_dict(torch.load('checkpoints/' + args.checkpoint + '.pth'))
    optimizer = optim.AdamW([
        {'params': list(test_model.ft.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': list(test_model.backbone.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5}
    ])
    criterion = (
        func.binary_cross_entropy_with_logits
        if task_type == "binclass"
        else func.cross_entropy
        if task_type == "multiclass"
        else func.mse_loss
    )
    
    with torch.no_grad():
        X_train_emb: list[torch.Tensor] = []
        X_test_emb: list[torch.Tensor] = []
        for i in range(math.ceil(X_train.shape[0]/args.batch)):
            X_train_emb.append(test_model(X_train[i * args.batch : (i + 1) * args.batch]))
        for i in range(math.ceil(X_test.shape[0]/args.batch)):
            X_test_emb.append(test_model(X_test[i * args.batch : (i + 1) * args.batch]))
        X_train_emb = torch.cat(X_train_emb, dim=0)
        X_test_emb = torch.cat(X_test_emb, dim=0)
        print(f'{X_train_emb.shape=}, {X_test_emb.shape=}')
        
        # LR = LinearRegression()
        # LR.fit(X_train_emb, Y_train)
        # LR_pred = LR.predict(X_test_emb)
        # score_LR = Evaluate(task_type, LR_pred, Y_test)  # equal to LeastSquare score
        
        Y_test_pred = LeastSquare(X_train_emb, Y_train, X_test_emb).detach().numpy()
        score_before = Evaluate(task_type, Y_test_pred, Y_test)
        
        print(f'Score before fine tuning: {score_before:.7f}.')
        with open("logs\log-XTab.txt", "a") as f:
            print(f'Score before fine tuning: {score_before:.7f}.', file=f)
    
    n_epochs = 1000000
    patience = 8
    batch_size = args.batch

    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(patience, mode="max")
    best_score = -math.inf
    best_epoch = 0
    
    timer.run()
    for epoch in range(n_epochs):
        for batch in tqdm(
            delu.iter_batches({'x': X_train, 'y': Y_train}, batch_size=batch_size, shuffle=True, drop_last=False),
            desc=f"Epoch {epoch}",
            total=math.ceil(X_train.shape[0]/batch_size),
            ncols=80
        ):
            test_model.train()
            optimizer.zero_grad()
            X_emb = test_model(batch['x'])
            y_pred = LeastSquare(X_emb, batch['y'], X_emb)
            loss = criterion(y_pred, batch['y'])
            loss.backward()
            optimizer.step()

        test_model.eval()
        
        with torch.no_grad():
            X_train_emb: list[torch.Tensor] = []
            X_test_emb: list[torch.Tensor] = []
            for i in range(math.ceil(X_train.shape[0]/args.batch)):
                X_train_emb.append(test_model(X_train[i * args.batch : (i + 1) * args.batch]))
            for i in range(math.ceil(X_test.shape[0]/args.batch)):
                X_test_emb.append(test_model(X_test[i * args.batch : (i + 1) * args.batch]))
            X_train_emb = torch.cat(X_train_emb, dim=0)
            X_test_emb = torch.cat(X_test_emb, dim=0)
            y_pred = LeastSquare(X_train_emb, Y_train, X_test_emb).detach().numpy()
            y_true = Y_test.detach().numpy()
            
            score = Evaluate(task_type, y_pred, y_true)
            
            log = f" [score] {score:.7f}  [time] {timer}"
            if score > best_score:
                best_score = score
                best_epoch = epoch
                log = '🌸' + log
            else:
                log = '  ' + log

            print(log)

        early_stopping.update(score)
        if epoch >= 10 and early_stopping.should_stop():
            break

    print("\n\nResult:")
    print('best =', best_score, 'epoch =', best_epoch)
    with open("logs\log-XTab.txt", "a") as f:
        print(datetime.now(), file=f)
        print(args, file=f)
        print('best =', best_score, 'epoch =', best_epoch, file=f)
        print('', file=f)