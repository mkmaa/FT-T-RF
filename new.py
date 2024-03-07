import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import copy
import tqdm
import delu
import math
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from read_data import read_XTab_dataset_train, read_XTab_dataset_test
from rtdl_revisiting_models import FTTransformer, FTTransformerBackbone

class RandomFeature(nn.Module):
    def __init__(self, n_features: int, d_embedding: int, n_dims: int):
        super(RandomFeature, self).__init__()
        self.d_embedding = d_embedding
        self.n_dims = n_dims
        self.clip_data_value = 27.6041
        
        rf_linear = nn.Linear(n_features, self.d_embedding, bias=True, dtype=torch.float32) # random feature
        nn.init.kaiming_normal_(rf_linear.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(rf_linear.bias)
        rf_linear.weight.requires_grad = False
        rf_linear.bias.requires_grad = False
        self.rf = nn.Sequential(rf_linear, nn.ReLU())
        self.pca = PCA(n_components=self.n_dims)

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        with torch.no_grad():
            x = self.rf(x)
        x = torch.from_numpy(self.pca.fit_transform(x.cpu().numpy())).to(x.device)
        x = torch.clamp(x, -self.clip_data_value, self.clip_data_value)
        x = x.unsqueeze(1)
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
    def __init__(self, num_datasets, n_features):
        super(Model, self).__init__()
        self.num_datasets = num_datasets
        self.rf = nn.ModuleList([RandomFeature(n_features=n_features[i], d_embedding=65536, n_dims=384) for i in range(num_datasets)])
        kwargs = FTTransformer.get_default_kwargs(n_blocks=6)
        del kwargs['_is_default']
        kwargs['d_out'] = None # 1, or when multiclass, should be set to n_classes
        self.backbone = FTTransformerBackbone(**kwargs)
        self.contrastive = nn.ModuleList([ContrastiveHead(d_in=384, d_out=n_features[i]) for i in range(num_datasets)])
        self.supervised = nn.ModuleList([SupervisedHead(d_in=384, d_out=1) for i in range(num_datasets)])

    def forward(self, x: dict[torch.Tensor]):
        contrast: list[torch.Tensor] = []
        prediction: list[torch.Tensor] = []
        for i in range(self.num_datasets):
            x[i] = self.rf[i](x[i])
            # print('rf =', x[i].shape)
            x[i] = self.backbone(x[i])
            # print('backbone =', x[i].shape)
            contrast.append(self.contrastive[i](x[i]))
            prediction.append(self.supervised[i](x[i]))
        return contrast, prediction
        
def train(args):
    dataX: list[torch.Tensor] = []
    dataY: list[torch.Tensor] = []
    n_features: list[int] = []
    n_samples: int = 0
    for dataset in args.training_dataset:
        print('Loading dataset', dataset, '...')
        task_type, X, Y, n = read_XTab_dataset_train(dataset)
        if task_type == 'regression':
            dataX.append(X)
            dataY.append(Y)
            n_features.append(n)
            n_samples += X.shape[0]
            print('| Loaded,', X.shape[0], 'samples,', X.shape[1], 'features.')
        else:
            print('| Skipped.')
    if len(n_features) != len(dataX) or len(n_features) != len(dataY):
        raise AssertionError('the size of data are different')
    num_datasets: int = len(n_features)
    print(num_datasets, 'datasets loaded,', n_samples, 'samples in total.')
    
    model = Model(num_datasets, n_features)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.Adam([
    #     {'params': list(test_model.backbone.parameters()), 'lr': 0.00001},
    #     {'params': list(test_model.contrastive.parameters()), 'lr': 0.00001},
    #     {'params': list(test_model.supervised.parameters()), 'lr': 0.00001}
    # ])
    criterion = nn.MSELoss()
    timer = delu.tools.Timer()
    
    model.train()
    timer.run()
    for epoch in range(50):
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
    task_type, X_train, X_test, Y_train, Y_test, n = read_XTab_dataset_test('__public__\\Laptop_Prices_Dataset')
    #                                                # __public__\\Laptop_Prices_Dataset abalone
    if task_type != 'regression':
        raise AssertionError('not regression')
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_model = Model(num_datasets=1, n_features=[n])
    # optimizer = optim.Adam(test_model.parameters(), lr=0.00001)
    optimizer = optim.Adam([
        {'params': list(test_model.backbone.parameters()), 'lr': 0.0001},
        # {'params': list(test_model.contrastive.parameters()), 'lr': 0.00001},
        {'params': list(test_model.supervised.parameters()), 'lr': 0.0001}
    ])
    
    criterion = nn.MSELoss()
    
    test_model.backbone.load_state_dict(torch.load('checkpoints/trained_backbone.pth'))

    # test_model.train()
    # best_score = None
    # best_epoch = None
    # for epoch in range(50):
    #     optimizer.zero_grad()
    #     data = copy.deepcopy([X])
        
    #     prediction: list[torch.Tensor]
    #     _, prediction = test_model(data)
    #     prediction[0] = prediction[0].squeeze(dim=1)
    #     loss = criterion(prediction[0], Y)
        
    #     score = -(mean_squared_error(Y.detach().numpy(), prediction[0].detach().numpy()) ** 0.5 * Y.std().item())
    #     if best_score is None or score > best_score:
    #         best_score = score
    #         best_epoch = epoch
        
    #     print('epoch =', epoch)
    #     print('| loss =', loss)
    #     print('| score =', score)

    #     loss.backward()
    #     optimizer.step()
    # print('best =', best_score)
    # print('epoch =', best_epoch)
    
    n_epochs = 1000000
    patience = 16

    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(patience, mode="max")
    best_score = -math.inf
    best_epoch = 0
    
    Y_std = Y_test.numpy().std()
    
    timer.run()
    for epoch in range(n_epochs):
        test_model.train()
        optimizer.zero_grad()
        data = copy.deepcopy([X_train])
        prediction: list[torch.Tensor]
        _, prediction = test_model(data)
        prediction[0] = prediction[0].squeeze(dim=1)
        loss = criterion(prediction[0], Y_train)
        
        loss.backward()
        optimizer.step()

        test_model.eval()
        test_data = copy.deepcopy([X_test])
        test_prediction: list[torch.Tensor]
        _, test_prediction = test_model(test_data)
        test_prediction[0] = test_prediction[0].squeeze(dim=1)
        score = -(mean_squared_error(test_prediction[0].detach().numpy(), Y_test.detach().numpy()) ** 0.5 * Y_std)
        
        print(f"score = {score:.4f} epoch = {epoch} [time] {timer}")

        early_stopping.update(score)
        if epoch >= 10 and early_stopping.should_stop():
            break

        if score > best_score:
            print("🌸 New best epoch! 🌸")
            best_score = score
            best_epoch = epoch

    print("\n\nResult:")
    print('best =', best_score, 'epoch =', best_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input args')
    # parser.add_argument('--dataset', type=str, choices=['CA', 'CO', 'KD', 'RC', 'NE', 'LF', 'DI'], help='dataset')
    # parser.add_argument('--rf_in', type=str, choices=['True', 'False'], help='inner random feature, replace feature tokenizer')
    # parser.add_argument('--rf_out', type=str, choices=['True', 'False'], help='outler random feature, work with feature tokenizer')
    # parser.add_argument('--bias', type=str, choices=['True', 'False'], help='bias, only in uniform init')
    # parser.add_argument('--activation', type=str, default='ReGLU', choices=['ReGLU', 'ReLU'], help='ReGLU or ReLU')
    # parser.add_argument('--init', type=str, choices=['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'], help='Xavier or Kaiming, uniform or normal')
    # parser.add_argument('--fan', type=str, choices=['fan_in', 'fan_out'], help='fan_in or fan_out, only in Kaiming init')
    # parser.add_argument('--n_blocks', type=int, default=3, choices=[1, 2, 3, 4, 5, 6], help='n_blocks of token embedding in FT-T')
    
    args = parser.parse_args()
    # if args.init == 'kaiming_uniform' or 'kaiming_normal':
    #     if args.fan == None:
    #         raise AssertionError('please input fan mode')
    
    args.training_dataset = [
        'abalone','Abalone_Dataset','ada_agnostic','Airfoil_Self-Noise','airfoil_self_noise','banknote_authentication','Bank_Customer_Churn_Dataset','Basketball_c','Bias_correction_of_numerical_prediction_model_temperature_forecast','Bias_correction_r','Bias_correction_r_2','bias_correction_ucl','BLE_RSSI_dataset_for_Indoor_localization','blogfeedback','BNG','c130','c131','c6','c7','c8','CDC_Diabetes_Health_Indicators','churn','communities_and_crime','Communities_and_Crime_Unnormalized','company_bankruptcy_prediction','cpmp-2015','cpu_small','Credit_c','Customer_Personality_Analysis','customer_satisfaction_in_airline','dabetes_130-us_hospitals','Data_Science_for_Good_Kiva_Crowdfunding','Data_Science_Salaries','delta_elevators','Diamonds','drug_consumption','dry_bean_dataset','E-CommereShippingData','eeg_eye_state','Facebook_Comment_Volume','Facebook_Comment_Volume_','Firm-Teacher_Clave-Direction_Classification','Fitness Club_c','Food_Delivery_Time','Gender_Gap_in_Spanish_WP','gina_agnostic','golf_play_dataset_extended','Healthcare_Insurance','Heart_Failure_Prediction','HR_Analytics_Job_Change_of_Data_Scientists','IBM_HR_Analytics_Employee_Attrition_and_Performance','in-vehicle-coupon-recommendation','INNHotelsGroup','insurance','irish','jm1','kc1','kc2','Large-scale_Wave_Energy_Farm_Perth_100','Large-scale_Wave_Energy_Farm_Perth_49','Large-scale_Wave_Energy_Farm_Sydney_100','Large-scale_Wave_Energy_Farm_Sydney_49','letter_recognition','maternal_health_risk','mice_protein_expression','Mobile_Phone_Market_in_Ghana','NHANES_age_prediction','obesity_estimation','objectivity_analysis','Parkinson_Multiple_Sound_Recording','pbc','pc1','pc3','pc4','phoneme','Physicochemical_Properties_of_Protein_Tertiary_Structure','Physicochemical_r','Pima_Indians_Diabetes_Database','productivity_prediction','qsar_aquatic_toxicity','QSAR_biodegradation','r29','r30','r36','Rain_in_Australia','rice_cammeo_and_osmancik','sensory','Smoking_and_Drinking_Dataset_with_body_signal','steel_industry_data','steel_industry_energy_consumption','steel_plates_faults','stock','Student_Alcohol_Consumption','superconductivity','Superconductivty','synchronous_machine','Telecom_Churn_Dataset','topo_2_1','turiye_student_evaluation','UJIndoorLoc','UJI_Pen_Characters','wave_energy_farm','Website_Phishing','Wine_Quality_','Wine_Quality_red','Wine_Quality_white','yeast'
    ]
    
    torch.manual_seed(0)
    delu.random.seed(0)
    
    # train(args)
    test(args)