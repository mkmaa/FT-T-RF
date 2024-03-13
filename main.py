import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import argparse
import copy
from tqdm.std import tqdm
import delu
import math
from datetime import datetime
import sklearn
from sklearn.decomposition import PCA

from read_data import read_XTab_dataset_train, read_XTab_dataset_test
from rtdl_revisiting_models import FTTransformer, FTTransformerBackbone, _CLSEmbedding
from loss import NTXent


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
        score = sklearn.metrics.accuracy_score(y_true, y_pred)
    elif task_type == "multiclass":
        y_pred = y_pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y_true, y_pred)
    else:
        assert task_type == "regression"
        score = -(sklearn.metrics.mean_squared_error(y_true, y_pred))
    return score


def LeastSquare(X_train, Y_train, X_test, alpha=0.0):
    X_train = torch.cat((X_train, torch.ones(X_train.shape[0], 1)), dim=1)
    X_test = torch.cat((X_test, torch.ones(X_test.shape[0], 1)), dim=1)
    I = torch.eye(X_train.shape[1])
    X_pinv = torch.inverse(X_train.t() @ X_train + alpha * I) @ X_train.t()
    theta = X_pinv @ Y_train
    Y_pred = X_test @ theta
    return Y_pred


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


class ReconstructionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super(ReconstructionHead, self).__init__()
        self.normalization = nn.LayerNorm(d_in)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_in, d_in, bias)
        self.linear2 = nn.Linear(d_in, d_out, bias)

    def forward(self, x):
        # x = x[:, :-1]
        x = self.linear1(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class ContrastiveHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super(ContrastiveHead, self).__init__()
        self.normalization = nn.LayerNorm(d_in)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_in, d_in, bias)
        self.linear2 = nn.Linear(d_in, d_out, bias)

    def forward(self, x):
        # x = x[:, :-1]
        x = self.linear1(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class SupervisedHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super(SupervisedHead, self).__init__()
        self.normalization = nn.LayerNorm(d_in)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_in, d_in, bias)
        self.linear2 = nn.Linear(d_in, d_out, bias)

    def forward(self, x):
        # x = x[:, -1]
        x = self.linear1(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Model(nn.Module):
    def __init__(self, num_datasets, n_features_list, args):
        super(Model, self).__init__()
        if args.mode == 'test':
            if num_datasets != 1:
                raise AssertionError('testing multiple datasets')
        self.num_datasets = num_datasets
        self.ft = nn.ModuleList([
            FeatureTokenizer(n_features=args.n_pca, n_dims=args.n_dims) for _ in range(num_datasets)
        ])
        kwargs = FTTransformer.get_default_kwargs(n_blocks=args.n_blocks)
        del kwargs['_is_default']
        kwargs['d_out'] = None # 1, or when multiclass, should be set to n_classes
        self.backbone = FTTransformerBackbone(**kwargs)
        self.reconstruction = nn.ModuleList([
            ReconstructionHead(d_in=args.n_dims, d_out=n_features_list[i]) for i in range(num_datasets)
        ]) if args.mode == 'train' else None
        self.contrastive = nn.ModuleList([
            ContrastiveHead(d_in=args.n_dims, d_out=n_features_list[i]) for i in range(num_datasets)
        ]) if args.mode == 'train' else None
        self.supervised = nn.ModuleList([
            SupervisedHead(d_in=args.n_dims, d_out=1) for _ in range(num_datasets)
        ])

    def forward(self, x):
        reconstruction: list[torch.Tensor] = []
        contrast: list[torch.Tensor] = []
        prediction: list[torch.Tensor] = []
        for i in range(self.num_datasets):
            x[i] = self.ft[i](x[i])
            # print('rf shape =', x[i].shape)
            x[i] = self.backbone(x[i])
            # print('backbone shape =', x[i].shape)
            if self.contrastive != None:
                reconstruction.append(self.reconstruction[i](x[i]))
                contrast.append(self.contrastive[i](x[i]))
            prediction.append(self.supervised[i](x[i]))
        return reconstruction, contrast, prediction
    

class Model_leastsq(nn.Module):
    def __init__(self, num_datasets, n_features_list, args):
        super(Model_leastsq, self).__init__()
        if args.mode != 'test' or num_datasets != 1:
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


def train(args):
    task_type: list[str] = []
    dataX: list[torch.Tensor] = []
    dataY: list[torch.Tensor] = []
    n_features_list: list[int] = []
    n_samples: int = 0
    for dataset in args.training_dataset:
        print('Loading dataset', dataset, '...')
        t, X, Y, n = read_XTab_dataset_train(dataset)
        if (t == 'binclass' or t == 'regression') and X.shape[1] <= 4:
            X, _ = RandomFeaturePreprocess(X, torch.Tensor(), d_embedding=args.d_embedding, n_dims=args.n_pca)
            task_type.append(t)
            dataX.append(X)
            dataY.append(Y)
            n_features_list.append(args.n_pca)
            n_samples += X.shape[0]
            print('| Loaded,', X.shape[0], 'samples,', X.shape[1], 'features.')
        else:
            print('| Skipped.')
    if len(n_features_list) != len(dataX) or len(n_features_list) != len(dataY):
        raise AssertionError('the size of data are different')
    num_datasets: int = len(n_features_list)
    print(num_datasets, 'datasets loaded,', n_samples, 'samples in total.')
    
    model = Model(num_datasets, n_features_list, args)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    ReconstructionLoss = func.mse_loss
    ContrastiveLoss = NTXent()
    SupervisedLoss = {'binclass': func.binary_cross_entropy_with_logits, 'regression': func.mse_loss}
    # timer = delu.tools.Timer()
    start_time = datetime.now()
    
    if args.load_checkpoint == 'True':
        checkpoint = torch.load('checkpoints/checkpoint.tar')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    
    model.train()
    # timer.run()
    for epoch in range(3):
        optimizer.zero_grad()
        data = copy.deepcopy(dataX)
        
        reconstruction: list[torch.Tensor] = []
        contrast: list[torch.Tensor]
        prediction: list[torch.Tensor]
        reconstruction, contrast, prediction = model(data)
        for i in range(num_datasets):
            prediction[i] = prediction[i].squeeze(dim=1)
        
        reconstruction_loss = sum(ReconstructionLoss(reconstruction[i], dataX[i]) for i in range(num_datasets))
        contrastive_loss = sum(ContrastiveLoss(contrast[i], dataX[i]) for i in range(num_datasets))
        supervised_loss = sum(SupervisedLoss[task_type[i]](prediction[i], dataY[i]) for i in range(num_datasets))
        total_loss = reconstruction_loss + contrastive_loss + supervised_loss
        
        print(f'epoch = {epoch + start_epoch}')
        print(f'| reconstruction loss = {reconstruction_loss.item():.7f}')
        print(f'| contrastive    loss = {contrastive_loss.item():.7f}')
        print(f'| supervisd      loss = {supervised_loss.item():.7f}')
        print(f'| total loss = {total_loss.item():.7f}')
        print(f'| [time] {datetime.now() - start_time}')

        total_loss.backward()
        optimizer.step()
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + start_epoch}
        torch.save(state, 'checkpoints/checkpoint.tar')
        torch.save(model.backbone.state_dict(), 'checkpoints/checkpoint_backbone.pth')
    torch.save(model.backbone.state_dict(), 'checkpoints/trained_backbone.pth')
    for i in range(num_datasets):
        torch.save(model.supervised[i].state_dict(), f'checkpoints/headers/{i}.pth')


def test(args):
    task_type, X_train, X_test, Y_train, Y_test, n = read_XTab_dataset_test('__public__\\' + args.dataset)
    print(f'Started training. Training size: {X_train.shape[0]}, testing size: {X_test.shape[0]}, feature number: {X_train.shape[1]}.')
    X_train, X_test = RandomFeaturePreprocess(X_train, X_test, d_embedding=args.d_embedding, n_dims=args.n_pca)
    print(f'Started training. After RF. Training size: {X_train.shape[0]}, testing size: {X_test.shape[0]}, feature number: {X_train.shape[1]}.')
    
    if task_type != 'regression' and task_type != 'binclass':
        raise AssertionError('not regression or binclass')
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'device: {device}')
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    Y_train = Y_train.to(device)
    
    test_model = Model(num_datasets=1, n_features_list=[n], args=args).to(device)
    optimizer = optim.AdamW([
        {'params': list(test_model.ft.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': list(test_model.backbone.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': list(test_model.supervised.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5}
    ])
    criterion = (
        func.binary_cross_entropy_with_logits
        if task_type == "binclass"
        else func.cross_entropy
        if task_type == "multiclass"
        else func.mse_loss
    )
    
    if args.pretrain == 'True':
        test_model.backbone.load_state_dict(torch.load('checkpoints\\' + args.checkpoint + '.pth'))
    
    n_epochs = 1000000
    patience = 32
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
            _, _, prediction = test_model([batch['x']])
            loss = criterion(prediction[0].squeeze(dim=1), batch['y'])
            loss.backward()
            optimizer.step()

        test_model.eval()
        _, _, test_prediction = test_model([X_test])
        
        y_pred = test_prediction[0].squeeze(dim=1).to('cpu').detach().numpy()
        y_true = Y_test.detach().numpy()
        
        # if task_type == "binclass":
        #     y_pred = np.round(scipy.special.expit(y_pred))
        #     score = sklearn.metrics.accuracy_score(y_true, y_pred)
        # elif task_type == "multiclass":
        #     y_pred = y_pred.argmax(1)
        #     score = sklearn.metrics.accuracy_score(y_true, y_pred)
        # else:
        #     assert task_type == "regression"
        #     score = -(sklearn.metrics.mean_squared_error(y_true, y_pred))
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
    with open("log-XTab.txt", "a") as f:
        print(datetime.now(), file=f)
        print(args, file=f)
        print('best =', best_score, 'epoch =', best_epoch, file=f)
        print('', file=f)


def test_leastsq(args):
    task_type, X_train, X_test, Y_train, Y_test, n = read_XTab_dataset_test('__public__\\' + args.dataset)
    print(f'Started training. Training size: {X_train.shape[0]}, testing size: {X_test.shape[0]}, feature number: {X_train.shape[1]}.')
    X_train, X_test = RandomFeaturePreprocess(X_train, X_test, d_embedding=args.d_embedding, n_dims=args.n_pca)
    print(f'Started training. After RF. Training size: {X_train.shape[0]}, testing size: {X_test.shape[0]}, feature number: {X_train.shape[1]}.')
    
    if task_type != 'regression' and task_type != 'binclass':
        raise AssertionError('not regression or binclass')
    
    if task_type == 'binclass':
        Y_train = Y_train * 2 - 1
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'device: {device}')
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    Y_train = Y_train.to(device)
    
    test_model = Model_leastsq(num_datasets=1, n_features_list=[n], args=args).to(device)
    test_model.backbone.load_state_dict(torch.load('checkpoints\\' + args.checkpoint + '.pth'))
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
        for i in tqdm(range(math.ceil(X_train.shape[0]/args.batch))):
            X_train_emb.append(test_model(X_train[i * args.batch : (i + 1) * args.batch]))
        for i in tqdm(range(math.ceil(X_test.shape[0]/args.batch))):
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
        with open("log-XTab.txt", "a") as f:
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
    with open("log-XTab.txt", "a") as f:
        print(datetime.now(), file=f)
        print(args, file=f)
        print('best =', best_score, 'epoch =', best_epoch, file=f)
        print('', file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input args')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'], help='training or testing')
    parser.add_argument('--dataset', type=str, help='choose the dataset')
    parser.add_argument('--pretrain', type=str, default='False', choices=['True', 'False'], help='whether to use the pretrained value')
    parser.add_argument('--checkpoint', type=str, default='trained_backbone', help='pretrained checkpoint')
    parser.add_argument('--load_checkpoint', type=str, default='False', choices=['True', 'False'], help='continue to train')
    parser.add_argument('--d_embedding', type=int, default=8192, help='embedding dim in RF')
    parser.add_argument('--n_pca', type=int, default=96, help='pca dim')
    parser.add_argument('--n_blocks', type=int, default=3, choices=[1, 2, 3, 4, 5, 6], help='n_blocks in PCA and FT-T')
    parser.add_argument('--batch', type=int, default=256, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    
    args = parser.parse_args()
    args.n_dims = [96, 128, 192, 256, 320, 384][args.n_blocks - 1]
    if args.mode == 'train':
        args.training_dataset = [
            'abalone','Abalone_Dataset','ada_agnostic','Airfoil_Self-Noise','airfoil_self_noise','banknote_authentication','Bank_Customer_Churn_Dataset','Basketball_c','Bias_correction_of_numerical_prediction_model_temperature_forecast','Bias_correction_r','Bias_correction_r_2','bias_correction_ucl','BLE_RSSI_dataset_for_Indoor_localization','blogfeedback','BNG','c130','c131','c6','c7','c8','CDC_Diabetes_Health_Indicators','churn','communities_and_crime','Communities_and_Crime_Unnormalized','company_bankruptcy_prediction','cpmp-2015','cpu_small','Credit_c','Customer_Personality_Analysis','customer_satisfaction_in_airline','dabetes_130-us_hospitals','Data_Science_for_Good_Kiva_Crowdfunding','Data_Science_Salaries','delta_elevators','Diamonds','drug_consumption','dry_bean_dataset','E-CommereShippingData','eeg_eye_state','Facebook_Comment_Volume','Facebook_Comment_Volume_','Firm-Teacher_Clave-Direction_Classification','Fitness Club_c','Food_Delivery_Time','Gender_Gap_in_Spanish_WP','gina_agnostic','golf_play_dataset_extended','Healthcare_Insurance','Heart_Failure_Prediction','HR_Analytics_Job_Change_of_Data_Scientists','IBM_HR_Analytics_Employee_Attrition_and_Performance','in-vehicle-coupon-recommendation','INNHotelsGroup','insurance','irish','jm1','kc1','kc2','Large-scale_Wave_Energy_Farm_Perth_100','Large-scale_Wave_Energy_Farm_Perth_49','Large-scale_Wave_Energy_Farm_Sydney_100','Large-scale_Wave_Energy_Farm_Sydney_49','letter_recognition','maternal_health_risk','mice_protein_expression','Mobile_Phone_Market_in_Ghana','NHANES_age_prediction','obesity_estimation','objectivity_analysis','Parkinson_Multiple_Sound_Recording','pbc','pc1','pc3','pc4','phoneme','Physicochemical_Properties_of_Protein_Tertiary_Structure','Physicochemical_r','Pima_Indians_Diabetes_Database','productivity_prediction','qsar_aquatic_toxicity','QSAR_biodegradation','r29','r30','r36','Rain_in_Australia','rice_cammeo_and_osmancik','sensory','Smoking_and_Drinking_Dataset_with_body_signal','steel_industry_data','steel_industry_energy_consumption','steel_plates_faults','stock','Student_Alcohol_Consumption','superconductivity','Superconductivty','synchronous_machine','Telecom_Churn_Dataset','topo_2_1','turiye_student_evaluation','UJIndoorLoc','UJI_Pen_Characters','wave_energy_farm','Website_Phishing','Wine_Quality_','Wine_Quality_red','Wine_Quality_white','yeast'
        ]
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    delu.random.seed(args.seed)
    
    if args.mode == 'train':
        train(args)
    else:
        # test(args)
        test_leastsq(args)