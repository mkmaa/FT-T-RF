import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.decomposition import PCA

from read_data import read_XTab_dataset
from rtdl_revisiting_models import FTTransformer, FTTransformerBackbone

class RandomFeature(nn.Module):
    def __init__(self, n_features: int, d_embedding: int = 65536, n_dims: int = 384):
        super(RandomFeature, self).__init__()
        # self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        # self.bias = nn.Parameter(torch.empty(n_features, d_embedding))
        # nn.init.kaiming_normal_(self.weight, 0, 'fan_out', 'relu')
        # nn.init.zeros_(self.bias)
        self.d_embedding = d_embedding
        self.n_dims = n_dims
        self.clip_data_value = 27.6041

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        rf_linear = nn.Linear(x.shape[1], self.d_embedding, bias=True, dtype=torch.float32) # random feature
        nn.init.kaiming_normal_(rf_linear.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(rf_linear.bias)
        rf_linear.weight.requires_grad = False
        rf_linear.bias.requires_grad = False
        rf = nn.Sequential(rf_linear, nn.ReLU()).to(x.device)
        with torch.no_grad():
            x = rf(x)
        self.pca = PCA(n_components=self.n_dims)
        x = torch.from_numpy(self.pca.fit_transform(x.cpu().numpy())).to(x.device)
        x = torch.clamp(x, -self.clip_data_value, self.clip_data_value)

class ContrastiveHead(nn.Module):
    def __init__(self, d_in: int = 384, d_out: int = 384, bias: bool = True):
        super(ContrastiveHead, self).__init__()
        self.normalization = nn.LayerNorm(d_in)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_in, d_in, bias)
        self.linear2 = nn.Linear(d_in, d_out, bias)

    def forward(self, x):
        x = x[:, :-1]
        x = self.linear1(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class SupervisedHead(nn.Module):
    def __init__(self, d_in: int = 384, d_out: int = 1, bias: bool = True):
        super(SupervisedHead, self).__init__()
        self.normalization = nn.LayerNorm(d_in)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_in, d_in, bias)
        self.linear2 = nn.Linear(d_in, d_out, bias)

    def forward(self, x):
        x = x[:, -1]
        x = self.linear1(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class Model(nn.Module):
    def __init__(self, num_datasets, n_features):
        super(Model, self).__init__()
        self.num_datasets = num_datasets
        self.rf = nn.ModuleList([RandomFeature(n_features[i]) for i in range(num_datasets)])
        kwargs = FTTransformer.get_default_kwargs(n_blocks=3)
        del kwargs['_is_default']
        kwargs['d_out'] = 1 # when multiclass, should be set to n_classes
        self.backbone = FTTransformerBackbone(**kwargs)
        self.contrastive = nn.ModuleList([ContrastiveHead() for _ in range(num_datasets)])
        self.supervised = nn.ModuleList([SupervisedHead() for _ in range(num_datasets)])

    def forward(self, datasets):
        contrast = []
        prediction = []
        for i in range(self.num_datasets):
            print('data =', datasets[i])
            rf = self.rf[i](datasets[i])
            print('rf =', rf)
            output = self.backbone(rf)
            contrast.append(self.contrastive[i](output))
            prediction.append(self.supervised[i](output))
        return contrast, prediction
        
def train(args):
    num_datasets = 1
    X_tot, Y_tot, n_feature = read_XTab_dataset('r1')
    X_tot = torch.from_numpy(X_tot).float()
    Y_tot = torch.from_numpy(Y_tot).float()
    dataX = [X_tot]  # Load your datasets
    dataY = [Y_tot]
    n_features = [n_feature]
    model = Model(num_datasets, n_features)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(100):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        contrast, prediction = model(dataX)

        # Calculate loss
        contrastive_loss = sum(criterion(contrast[i], dataX[i]) for i in range(num_datasets))
        supervised_loss = sum(criterion(prediction[i], dataY[i]) for i in range(num_datasets))
        total_loss = contrastive_loss + supervised_loss

        # Backward pass
        total_loss.backward()

        # Update the weights
        optimizer.step()
    torch.save(model.backbone.state_dict(), 'checkpoints/backbone.pth')
        
def test(args):
    # 初始化新的模型
    test_model = Model(num_datasets=1)  # 假设你有一个支持集

    # 加载主网络的权重
    test_model.backbone.load_state_dict(torch.load('checkpoints/backbone.pth'))

    # 将模型设置为评估模式
    test_model.eval()

    # 加载支持集
    support_set = [...]  # 加载你的支持集

    # 使用新的编码器和解码器，复用主网络的参数，进行前向传播
    decoded = test_model(support_set)
    
    criterion = nn.MSELoss()
    loss = criterion(decoded[0], support_set[0])
    print('loss =', loss)

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
    
    train(args)
    test(args)