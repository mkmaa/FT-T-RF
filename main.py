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
    # print(f'Before RF {X_train.shape=}, {X_test.shape=}.')
    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    X = torch.cat((X_train, X_test), dim=0)
    weight = torch.empty(X.shape[1], d_embedding)
    nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
    X = X[..., None] * weight
    X = torch.sum(X, dim=1)
    # print(f'{X.shape=}')
    X = nn.ReLU()(X)
    pca = PCA(n_components=n_dims)
    X = torch.from_numpy(pca.fit_transform(X.cpu().numpy())).to(X.device)
    # print(f'After RF {X.shape=}')
    return torch.split(X, [n_samples_train, n_samples_test], dim=0)


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
        self.num_datasets = num_datasets
        self.ft = nn.ModuleList([
            FeatureTokenizer(n_features=args.n_pca, n_dims=args.n_dims)for i in range(num_datasets)
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
            SupervisedHead(d_in=args.n_dims, d_out=1) for i in range(num_datasets)
        ])

    def forward(self, x: dict[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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


def train(args):
    dataX: list[torch.Tensor] = []
    dataY: list[torch.Tensor] = []
    n_features_list: list[int] = []
    n_samples: int = 0
    for dataset in args.training_dataset:
        print('Loading dataset', dataset, '...')
        task_type, X, Y, n = read_XTab_dataset_train(dataset)
        if task_type != 'multiclass' and X.shape[1] <= 8:
            X, _ = RandomFeaturePreprocess(X, torch.Tensor(), d_embedding=args.d_embedding, n_dims=args.n_pca)
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
    SupervisedLoss = (
        func.binary_cross_entropy_with_logits
        if task_type == "binclass"
        else func.cross_entropy
        if task_type == "multiclass"
        else func.mse_loss
    )
    timer = delu.tools.Timer()
    
    model.train()
    timer.run()
    for epoch in range(100):
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
        supervised_loss = sum(SupervisedLoss(prediction[i], dataY[i]) for i in range(num_datasets))
        total_loss = reconstruction_loss + contrastive_loss + supervised_loss
        
        print('epoch =', epoch)
        print('| reconstruction loss =', reconstruction_loss)
        print('| contrastive    loss =', contrastive_loss)
        print('| supervisd      loss =', supervised_loss)
        print('| total loss =', total_loss)
        print(f'| [time] {timer}')

        total_loss.backward()
        optimizer.step()
    torch.save(model.backbone.state_dict(), 'checkpoints/trained_backbone.pth')
    # torch.save(model.supervised[0].state_dict(), 'checkpoints/trained_header.pth')


def test(args):
    task_type, X_train, X_test, Y_train, Y_test, n = read_XTab_dataset_test('__public__\\' + args.dataset)
    #                                                # __public__\\Laptop_Prices_Dataset        abalone
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
        # {'params': list(test_model.contrastive.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5},
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
    patience = 16
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
            # print(f'{len(prediction)=}, {prediction[0].shape=}.')
            loss = criterion(prediction[0].squeeze(dim=1), batch['y'])
            # print(f'loss = {loss.item()}')
            loss.backward()
            optimizer.step()

        test_model.eval()
        _, _, test_prediction = test_model([X_test])
        
        y_pred = test_prediction[0].squeeze(dim=1).to('cpu').detach().numpy()
        y_true = Y_test.detach().numpy()
        
        if task_type == "binclass":
            y_pred = np.round(scipy.special.expit(y_pred))
            score = sklearn.metrics.accuracy_score(y_true, y_pred)
        elif task_type == "multiclass":
            y_pred = y_pred.argmax(1)
            score = sklearn.metrics.accuracy_score(y_true, y_pred)
        else:
            assert task_type == "regression"
            score = -(sklearn.metrics.mean_squared_error(y_true, y_pred))
        
        log = f" [score] {score:.4f}  [time] {timer}"

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
    parser.add_argument('--d_embedding', type=int, default=8192, help='embedding dim in RF')
    parser.add_argument('--n_pca', type=int, default=8, help='pca dim')
    parser.add_argument('--n_blocks', type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help='n_blocks in PCA and FT-T')
    parser.add_argument('--batch', type=int, default=256, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    
    args = parser.parse_args()
    args.n_dims = [96, 128, 192, 256, 320, 384][args.n_blocks - 1]
    if args.mode == 'train':
        args.training_dataset = [
            'abalone','Abalone_Dataset','ada_agnostic','Airfoil_Self-Noise','airfoil_self_noise','banknote_authentication','Bank_Customer_Churn_Dataset','Basketball_c','Bias_correction_of_numerical_prediction_model_temperature_forecast','Bias_correction_r','Bias_correction_r_2','bias_correction_ucl','BLE_RSSI_dataset_for_Indoor_localization','blogfeedback','BNG','c130','c131','c6','c7','c8','CDC_Diabetes_Health_Indicators','churn','communities_and_crime','Communities_and_Crime_Unnormalized','company_bankruptcy_prediction','cpmp-2015','cpu_small','Credit_c','Customer_Personality_Analysis','customer_satisfaction_in_airline','dabetes_130-us_hospitals','Data_Science_for_Good_Kiva_Crowdfunding','Data_Science_Salaries','delta_elevators','Diamonds','drug_consumption','dry_bean_dataset','E-CommereShippingData','eeg_eye_state','Facebook_Comment_Volume','Facebook_Comment_Volume_','Firm-Teacher_Clave-Direction_Classification','Fitness Club_c','Food_Delivery_Time','Gender_Gap_in_Spanish_WP','gina_agnostic','golf_play_dataset_extended','Healthcare_Insurance','Heart_Failure_Prediction','HR_Analytics_Job_Change_of_Data_Scientists','IBM_HR_Analytics_Employee_Attrition_and_Performance','in-vehicle-coupon-recommendation','INNHotelsGroup','insurance','irish','jm1','kc1','kc2','Large-scale_Wave_Energy_Farm_Perth_100','Large-scale_Wave_Energy_Farm_Perth_49','Large-scale_Wave_Energy_Farm_Sydney_100','Large-scale_Wave_Energy_Farm_Sydney_49','letter_recognition','maternal_health_risk','mice_protein_expression','Mobile_Phone_Market_in_Ghana','NHANES_age_prediction','obesity_estimation','objectivity_analysis','Parkinson_Multiple_Sound_Recording','pbc','pc1','pc3','pc4','phoneme','Physicochemical_Properties_of_Protein_Tertiary_Structure','Physicochemical_r','Pima_Indians_Diabetes_Database','productivity_prediction','qsar_aquatic_toxicity','QSAR_biodegradation','r29','r30','r36','Rain_in_Australia','rice_cammeo_and_osmancik','sensory','Smoking_and_Drinking_Dataset_with_body_signal','steel_industry_data','steel_industry_energy_consumption','steel_plates_faults','stock','Student_Alcohol_Consumption','superconductivity','Superconductivty','synchronous_machine','Telecom_Churn_Dataset','topo_2_1','turiye_student_evaluation','UJIndoorLoc','UJI_Pen_Characters','wave_energy_farm','Website_Phishing','Wine_Quality_','Wine_Quality_red','Wine_Quality_white','yeast'
        ]
    
    torch.manual_seed(args.seed)
    delu.random.seed(args.seed)
    
    if args.mode == 'train':
        train(args)
    else:
        test(args)