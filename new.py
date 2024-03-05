import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.decomposition import PCA
import copy

from read_data import read_XTab_dataset
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
    for dataset in args.training_dataset:
        X, Y, n = read_XTab_dataset(dataset)
        dataX.append(X)
        dataY.append(Y)
        n_features.append(n)
    if len(n_features) != len(dataX) or len(n_features) != len(dataY):
        raise AssertionError('the size of data are different')
    num_datasets: int = len(n_features)
    
    model = Model(num_datasets, n_features)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        data = copy.deepcopy(dataX)
        
        contrast: list[torch.Tensor]
        prediction: list[torch.Tensor]
        contrast, prediction = model(data)
        for i in range(num_datasets):
            prediction[i] = prediction[i].squeeze(dim=1)
        
        # print('contrast =', contrast[0].shape)
        # print('prediction =', prediction[0].shape)
        # print('dataX =', dataX[0].shape)
        # print('dataY =', dataY[0].shape)
        
        contrastive_loss = sum(criterion(contrast[i], dataX[i]) for i in range(num_datasets))
        supervised_loss = sum(criterion(prediction[i], dataY[i]) for i in range(num_datasets))
        total_loss = contrastive_loss + 2*supervised_loss
        
        print('epoch =', epoch)
        print('| contr loss =', contrastive_loss)
        print('| supvi loss =', supervised_loss)
        print('| total loss =', total_loss)

        total_loss.backward()
        optimizer.step()
    torch.save(model.backbone.state_dict(), 'checkpoints/trained_backbone.pth')
    # torch.save(model.supervised[0].state_dict(), 'checkpoints/trained_header.pth')
        
def test(args):
    X, Y, n = read_XTab_dataset('r1')
    test_model = Model(num_datasets=1, n_features=[n])
    optimizer = optim.Adam(test_model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    test_model.backbone.load_state_dict(torch.load('checkpoints/trained_backbone.pth'))

    test_model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        data = copy.deepcopy([X])
        
        prediction: list[torch.Tensor]
        _, prediction = test_model(data)
        prediction[0] = prediction[0].squeeze(dim=1)
        loss = criterion(prediction[0], Y)
        
        print('epoch =', epoch)
        print('| supvi loss =', loss)

        loss.backward()
        optimizer.step()

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
    
    args.training_dataset = ['r1', 'r2', 'r35']
    
    train(args)
    # test(args)