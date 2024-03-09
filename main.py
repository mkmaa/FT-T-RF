import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import copy
from tqdm.std import tqdm
import delu
import math
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from read_data import read_XTab_dataset_train, read_XTab_dataset_test
from rtdl_revisiting_models import FTTransformer, FTTransformerBackbone, _CLSEmbedding


class RandomFeature(nn.Module):
    def __init__(self, n_features: int, d_embedding: int, n_dims: int, n_ensemble: int):
        super(RandomFeature, self).__init__()
        self.d_embedding = d_embedding
        self.n_dims = n_dims
        self.clip_data_value = 27.6041
        # self.weight = nn.Parameter(torch.empty(n_features, d_embedding, rf_size))
        # self.bias = nn.Parameter(torch.empty(n_features, d_embedding, rf_size))
        
        self.cls = _CLSEmbedding(n_dims)
        self.rf = nn.ModuleList()
        self.pca = []
        for _ in range(n_ensemble):
            rf_linear = nn.Linear(n_features, self.d_embedding, bias=True, dtype=torch.float32) # random feature
            nn.init.kaiming_normal_(rf_linear.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(rf_linear.bias)
            rf_linear.weight.requires_grad = False
            rf_linear.bias.requires_grad = False
            self.rf.append(nn.Sequential(rf_linear, nn.ReLU()))
            self.pca.append(PCA(n_components=self.n_dims))
        
        # rf_linear = nn.Linear(n_features, self.d_embedding, bias=True, dtype=torch.float32) # random feature
        # nn.init.kaiming_normal_(rf_linear.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.zeros_(rf_linear.bias)
        # rf_linear.weight.requires_grad = False
        # rf_linear.bias.requires_grad = False
        # self.rf = nn.Sequential(rf_linear, nn.ReLU())
        # self.pca = PCA(n_components=self.n_dims)

    def forward(self, x: torch.Tensor):
        # with torch.no_grad():
        #     x = x[..., None, None] * self.weight
        #     x = x + self.bias[None]
        #     x = torch.sum(x, dim=1)
        #     x = x.permute(0, 2, 1)
        #     x = self.relu(x)
        # original_shape = x.shape
        # x = x.reshape(-1, original_shape[1])
        # self.pca = PCA(n_components=self.pca_size)
        # x = self.pca.fit_transform(x.cpu().numpy())
        # x = torch.from_numpy(x).to('cuda:0').reshape(original_shape[0], self.pca_size, original_shape[2])
        # # x = torch.from_numpy(self.pca.fit_transform(x.cpu().numpy())).to(x.device)
        # x = torch.clamp(x, -self.clip_data_value, self.clip_data_value)
        
        outputs = []
        x_cls = self.cls(x.shape[:-1]).squeeze(dim=1)
        # print(f'cls shape = {x_cls.shape}')
        outputs.append(x_cls)
        for rf, pca in zip(self.rf, self.pca):
            with torch.no_grad():
                x_rf = rf(x)
            x_pca = torch.from_numpy(pca.fit_transform(x_rf.cpu().numpy())).to(x.device)
            x_clamp = torch.clamp(x_pca, -self.clip_data_value, self.clip_data_value)
            # print(f'clamp shape = {x_clamp.shape}')
            outputs.append(x_clamp)
        return torch.stack(outputs, dim=1)
        
        # x = x.flatten(start_dim=1)
        # with torch.no_grad():
        #     x = self.rf(x)
        # x = torch.from_numpy(self.pca.fit_transform(x.cpu().numpy())).to(x.device)
        # x = torch.clamp(x, -self.clip_data_value, self.clip_data_value)
        # x = x.unsqueeze(1)
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
        self.rf = nn.ModuleList([
            RandomFeature(n_features=n_features_list[i], d_embedding=args.d_embedding, n_dims=args.n_dims, n_ensemble=args.n_ensemble)for i in range(num_datasets)
        ])
        kwargs = FTTransformer.get_default_kwargs(n_blocks=args.n_blocks)
        del kwargs['_is_default']
        kwargs['d_out'] = None # 1, or when multiclass, should be set to n_classes
        self.backbone = FTTransformerBackbone(**kwargs)
        self.contrastive = nn.ModuleList([
            ContrastiveHead(d_in=args.n_dims, d_out=n_features_list[i]) for i in range(num_datasets)
        ]) if args.mode == 'train' else None
        self.supervised = nn.ModuleList([
            SupervisedHead(d_in=args.n_dims, d_out=1) for i in range(num_datasets)
        ])

    def forward(self, x: dict[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        contrast: list[torch.Tensor] = []
        prediction: list[torch.Tensor] = []
        for i in range(self.num_datasets):
            x[i] = self.rf[i](x[i])
            # print('rf shape =', x[i].shape)
            x[i] = self.backbone(x[i])
            # print('backbone shape =', x[i].shape)
            if self.contrastive != None:
                contrast.append(self.contrastive[i](x[i]))
            prediction.append(self.supervised[i](x[i]))
        return contrast, prediction


def train(args):
    dataX: list[torch.Tensor] = []
    dataY: list[torch.Tensor] = []
    n_features_list: list[int] = []
    n_samples: int = 0
    for dataset in args.training_dataset:
        print('Loading dataset', dataset, '...')
        task_type, X, Y, n = read_XTab_dataset_train(dataset)
        if task_type == 'regression':
            dataX.append(X)
            dataY.append(Y)
            n_features_list.append(n)
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
    # optimizer = optim.Adam([
    #     {'params': list(test_model.backbone.parameters()), 'lr': 0.00001},
    #     {'params': list(test_model.contrastive.parameters()), 'lr': 0.00001},
    #     {'params': list(test_model.supervised.parameters()), 'lr': 0.00001}
    # ])
    criterion = nn.MSELoss()
    timer = delu.tools.Timer()
    
    model.train()
    timer.run()
    for epoch in range(100):
        optimizer.zero_grad()
        data = copy.deepcopy(dataX)
        
        contrast: list[torch.Tensor]
        prediction: list[torch.Tensor]
        contrast, prediction = model(data)
        for i in range(num_datasets):
            prediction[i] = prediction[i].squeeze(dim=1)
        
        contrastive_loss = sum(criterion(contrast[i], dataX[i]) for i in range(num_datasets))
        supervised_loss = sum(criterion(prediction[i], dataY[i]) for i in range(num_datasets))
        total_loss = contrastive_loss + supervised_loss
        
        print('epoch =', epoch)
        print('| contr loss =', contrastive_loss)
        print('| supvi loss =', supervised_loss)
        print('| total loss =', total_loss)
        print(f'| [time] {timer}')

        total_loss.backward()
        optimizer.step()
    torch.save(model.backbone.state_dict(), 'checkpoints/trained_backbone.pth')
    # torch.save(model.supervised[0].state_dict(), 'checkpoints/trained_header.pth')


def test(args):
    task_type, X_train, X_test, Y_train, Y_test, n = read_XTab_dataset_test('__public__\\' + args.dataset)
    #                                                # __public__\\Laptop_Prices_Dataset        abalone
    print(f'training size = {X_train.shape[0]}, testing size = {X_test.shape[0]}.')
    
    if task_type != 'regression':
        raise AssertionError('not regression')
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_model = Model(num_datasets=1, n_features_list=[n], args=args)
    # optimizer = optim.Adam(test_model.parameters(), lr=0.00001)
    # optimizer = optim.Adam([
    #     {'params': list(test_model.backbone.parameters()), 'lr': 0.0001},
    #     # {'params': list(test_model.contrastive.parameters()), 'lr': 0.00001},
    #     {'params': list(test_model.supervised.parameters()), 'lr': 0.0001}
    # ])
    optimizer = optim.AdamW([
        {'params': list(test_model.backbone.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5},
        # {'params': list(test_model.contrastive.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': list(test_model.supervised.parameters()), 'lr': 1e-4, 'weight_decay': 1e-5}
    ])
    
    criterion = nn.MSELoss()
    
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
            delu.iter_batches({'x': X_train, 'y': Y_train}, batch_size=batch_size, shuffle=True, drop_last=True),
            desc=f"Epoch {epoch}",
            total=math.floor(X_train.shape[0]/batch_size),
            ncols=80
        ):
            test_model.train()
            optimizer.zero_grad()
            _, prediction = test_model([batch['x']])
            # print(f'len pred = {len(prediction)}, shape pred = {prediction[0].shape}')
            loss = criterion(prediction[0].squeeze(dim=1), batch['y'])
            # print(f'loss = {loss.item()}')
            loss.backward()
            optimizer.step()

        test_model.eval()
        _, test_prediction = test_model([X_test])
        test_prediction[0] = test_prediction[0].squeeze(dim=1)
        score = -(mean_squared_error(test_prediction[0].detach().numpy(), Y_test.detach().numpy()))
        
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
    parser.add_argument('--n_ensemble', type=int, default=8, help='ensemble num of RF')
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