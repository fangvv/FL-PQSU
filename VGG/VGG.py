import math
import torch.nn as nn
import numpy as np

__all__ = ['vgg']

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

defaultFullLayers = [4096, 4096]

class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=16, init_weights=True, cfg=None, fullLayers=None):
        global num_classes
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        if fullLayers is None:
            fullLayers = defaultFullLayers

        self.cfg = cfg
        self.fullLayers = fullLayers

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], fullLayers[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fullLayers[0], fullLayers[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fullLayers[1], num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = vgg(depth=16,fullLayers=[4096,4096])
    list_name = []
    for name in net.state_dict():
        list_name.append(name)
    for name in list_name:
        temp_np = net.state_dict()[name].numpy()
        np.save("./SaveQuanOrgModel/%s.ndim" % (name), temp_np)
