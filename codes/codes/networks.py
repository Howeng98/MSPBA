import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from .utils import makedirpath

from torchvision import models

__all__ = ['Encoder64', 'Encoder32', 'Encoder16', 'PositionClassifier', 'Vgg16']


class Vgg16(nn.Module):
    def __init__(self, pretrained = True):
        super(Vgg16, self).__init__()
        self.vggnet = models.vgg16(pretrained)
        del(self.vggnet.classifier) # Remove fully connected layer to save memory.
        features = list(self.vggnet.features)
        self.layers = nn.ModuleList(features).eval() 
        
    def forward(self, x, size='large'):
        results = []
        if size == 'large':
            for ii,model in enumerate(self.layers):
                x = model(x)
                if ii in [3,8,15,22,29]:
                    results.append(x)
            return results
        else:
            for ii,model in enumerate(self.layers):
                if ii <= 15:
                    x = model(x)
                    if ii in [3,8,15]:
                        results.append(x)
            return results


class Encoder64(nn.Module):
    def __init__(self, pretrained_net, K, D=64, bias=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(512, 256, 2, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(256, 128, 2, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(128, 64, 2, 1, 0, bias=bias)
        self.conv4 = nn.Conv2d(64, D, 1, 1, 0, bias=bias)
        
        self.K = K
        self.D = D
        
        self.pretrained_net = pretrained_net

    def forward(self, x):
        h = self.pretrained_net(x)[4]
                
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h)
        h = F.leaky_relu(h, 0.1)
        
        h = self.conv4(h)
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/encoder_nohier.pkl'


class Encoder32(nn.Module):
    def __init__(self, pretrained_net, K, D=64, bias=True):
        super().__init__()      
        
        self.conv1 = nn.Conv2d(256, 256, 3, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(256, 128, 3, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(128, 128, 2, 1, 0, bias=bias)
        self.conv4 = nn.Conv2d(128, 64, 2, 1, 0, bias=bias)
        self.conv5 = nn.Conv2d(64, D, 2, 1, 0, bias=bias)
        
        self.K = K
        self.D = D
        
        self.pretrained_net = pretrained_net

    def forward(self, x):
        h = self.pretrained_net(x)[2]
        
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv4(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv5(h)
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/encdeep.pkl'


class Encoder16(nn.Module):
    def __init__(self, pretrained_net, K, D=64, bias=True):
        super().__init__()      
        
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 0, bias=bias)
        self.conv4 = nn.Conv2d(32, D, 2, 1, 0, bias=bias)
        
        self.K = K
        self.D = D
        
        self.pretrained_net = pretrained_net

    def forward(self, x):
        h = self.pretrained_net(x,'small')[1]
        
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = F.leaky_relu(h, 0.1)
        
        h = self.conv3(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv4(h)
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/encdeep.pkl'


################


xent = nn.CrossEntropyLoss()


class NormalizedLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        with torch.no_grad():
            w = self.weight / self.weight.data.norm(keepdim=True, dim=0)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class PositionClassifier(nn.Module):
    def __init__(self, K, D, class_num=12):
        super().__init__()
        self.D = D

        self.fc1 = nn.Linear(D, 128)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(128, 256)
        self.act2 = nn.LeakyReLU(0.1)
        
        self.fc3 = nn.Linear(256, 512)
        self.act3 = nn.LeakyReLU(0.1)

        self.fc4 = NormalizedLinear(512, class_num)

        self.K = K

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self, name):
        return f'ckpts/{name}/position_classifier_K{self.K}.pkl'

    @staticmethod
    def infer(c, enc, batch):
        x1s, x2s, ys = batch

        h1 = enc(x1s)
        h2 = enc(x2s)

        logits = c(h1, h2)
        loss = xent(logits, ys)
        return loss

    def forward(self, h1, h2):
        h1 = h1.view(-1, self.D)
        h2 = h2.view(-1, self.D)

        h = h1 - h2

        h = self.fc1(h)
        h = self.act1(h)

        h = self.fc2(h)
        h = self.act2(h)
        
        h = self.fc3(h)
        h = self.act3(h)

        h = self.fc4(h)
        return h

