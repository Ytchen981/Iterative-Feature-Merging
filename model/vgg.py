import torch
import torch.nn as nn
from utils.self_weight_matching import axes2perm_to_perm2axes, self_merge_weight


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, nclass=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(cfg[vgg_name][-2], nclass)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    @ staticmethod
    def get_axis_to_perm(model):
        next_perm = None
        axis_to_perm = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                previous_perm = next_perm
                next_perm = f"perm_{name}"
                axis_to_perm[f"{name}.weight"] = (next_perm, previous_perm, None, None)
                axis_to_perm[f"{name}.bias"] = (next_perm, None)
            elif isinstance(module, nn.BatchNorm2d):
                axis_to_perm[f"{name}.weight"] = (next_perm, None)
                axis_to_perm[f"{name}.bias"] = (next_perm, None)
                axis_to_perm[f"{name}.running_mean"] = (next_perm, None)
                axis_to_perm[f"{name}.running_var"] = (next_perm, None)
                axis_to_perm[f"{name}.num_batches_tracked"] = ()
            elif isinstance(module, nn.Linear):
                axis_to_perm[f"{name}.weight"] = (None, next_perm)
                axis_to_perm[f"{name}.bias"] = (None, )
        return axis_to_perm


class VGG_cluster(nn.Module):
    def __init__(self, vgg_name, channel_num):
        super(VGG_cluster, self).__init__()
        self.channel_num = channel_num
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(channel_num[-1], 10)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        i = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.channel_num[i] is None:
                    n_c = x
                else:
                    n_c = self.channel_num[i]
                layers += [nn.Conv2d(in_channels, n_c, kernel_size=3, padding=1),
                           nn.BatchNorm2d(n_c),
                           nn.ReLU(inplace=True)]
                in_channels = n_c
                i += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_no_act(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_no_act, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.Identity(inplace=True)
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    @ staticmethod
    def get_axis_to_perm(model):
        next_perm = None
        axis_to_perm = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                previous_perm = next_perm
                next_perm = f"perm_{name}"
                axis_to_perm[f"{name}.weight"] = (next_perm, previous_perm, None, None)
                axis_to_perm[f"{name}.bias"] = (next_perm, None)
            elif isinstance(module, nn.BatchNorm2d):
                axis_to_perm[f"{name}.weight"] = (next_perm, None)
                axis_to_perm[f"{name}.bias"] = (next_perm, None)
                axis_to_perm[f"{name}.running_mean"] = (next_perm, None)
                axis_to_perm[f"{name}.running_var"] = (next_perm, None)
                axis_to_perm[f"{name}.num_batches_tracked"] = ()
            elif isinstance(module, nn.Linear):
                axis_to_perm[f"{name}.weight"] = (None, next_perm)
                axis_to_perm[f"{name}.bias"] = (None)
        return axis_to_perm

def merge_channel_vgg16(vgg_name, model_param, max_ratio=0.5, threshold=0.1):
    model = VGG(vgg_name)
    axes_to_perm = VGG.get_axis_to_perm(model)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight(perm_to_axes, model_param, max_ratio, threshold)
    perms = perm_size.keys()
    #perms.sort()
    channel_num = []
    for key in perms:
        channel_num.append(perm_size[key])
    new_model = VGG_cluster(vgg_name, channel_num)
    new_model.load_state_dict(param)
    return new_model, perm_size


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

if __name__ == "__main__":
    model_names = ['VGG11']
    for model_name in model_names:
        model = VGG(model_name)
        layers_to_hook = []
        get_hook = False
        for name, module in model.named_modules():
            if get_hook:
                layers_to_hook.append(name)
                get_hook = False
            if isinstance(module, nn.Conv2d):
                mtype = "conv2d"
            elif isinstance(module, nn.BatchNorm2d):
                mtype = "bn2d"
            elif isinstance(module, nn.ReLU):
                mtype = "relu"
                get_hook = True
            elif isinstance(module, nn.MaxPool2d):
                mtype = "maxpool"
            else:
                mtype = "Unknown"
            # print(f"{name}:{mtype}")
        print(f"{model_name}: {layers_to_hook}")
