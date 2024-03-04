#from model.resnet_torch_flexi_new_mid import ResNet50, merge_channel_ResNet50
from model.vgg import merge_channel_vgg16, VGG
from utils.utils import flatten_params, lerp
from utils.plot import plot_interp_acc
from utils.activation_cluster import get_activation, get_activation_cluster, compare_activation
import argparse
import torch
import os
from torchvision import datasets, transforms
from utils.training import test
from tqdm import tqdm
import copy
from thop import profile
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from utils.utils import Cam_overlay, ImageNet_train_transform, ImageNet_test_transform

def test_activation_cluster(origin_model, checkpoint, dataloader, max_ratio, threshold, figure_path, vgg_name):
    '''origin_model.cuda()
    origin_model.eval()
    correct = 0
    total_num = 0
    total_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X.cuda()
            y = y.cuda()
            logit = origin_model(X)
            loss = F.cross_entropy(logit, y)
            correct += (logit.max(1)[1] == y).sum().item()
            total_num += y.size(0)
            total_loss += loss.item()
    print(f"origin model: acc:{correct / total_num * 100:.2f}%, avg loss:{total_loss / (i + 1):.4f}")'''

    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()
    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    model, _ = merge_channel_vgg16(vgg_name, checkpoint, max_ratio=max_ratio, threshold=threshold)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    correct = 0
    total_num = 0
    total_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X.cuda()
            y = y.cuda()
            logit = model(X)
            loss = F.cross_entropy(logit, y)
            correct += (logit.max(1)[1] == y).sum().item()
            total_num += y.size(0)
            total_loss += loss.item()
    print(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}")
    print(
        f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    with open(os.path.join(figure_path, "self_merge_result_mid.txt"), 'a+') as f:
        f.write(f"max_ratio:{max_ratio}, threshold:{threshold}\n")
        f.write(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}\n")
        f.write(f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %\n\n")
    return 0

def get_AVG_CAM(origin_model, activation_cluster, checkpoint, dataloader, fig_path):
    if not os.path.exists(os.path.join(fig_path, "avg_cam")):
        os.mkdir(os.path.join(fig_path, "avg_cam"))
    origin_model.load_state_dict(checkpoint)
    origin_model.cuda()
    origin_model.eval()
    feature_map = {}
    def get_feature(name):
        def pre_hook(module, data_input):
            with torch.no_grad():
                input = data_input[0]
                if name not in feature_map.keys():
                    feature_map[name] = input
                else:
                    feature_map[name] += input
        return pre_hook

    hooks = []
    for name, module in origin_model.named_modules():
        if name in activation_cluster.keys():
            h = module.register_forward_pre_hook(get_feature(name))
            hooks.append(h)
    for i, (X, y) in enumerate(dataloader):
        with torch.no_grad():
            X = X.cuda()
            _ = origin_model(X)
        break
    for h in hooks:
        h.remove()

    if not os.path.exists(os.path.join(fig_path, "avg_cam", "origin")):
        os.mkdir(os.path.join(fig_path, "avg_cam", "origin"))
    for k in range(X.shape[0]):
        origin_img = X[k].cpu().squeeze()
        img_path = os.path.join(fig_path, "avg_cam", "origin", f"{k}.png")
        save_image(origin_img, img_path)
    for name in tqdm(activation_cluster.keys()):
        if name == "features.27":
            break
        if not os.path.exists(os.path.join(fig_path, "avg_cam", f"{name}")):
            os.mkdir(os.path.join(fig_path, "avg_cam", f"{name}"))
        for key in activation_cluster[name].keys():
            if key >=0:
                if not os.path.exists(os.path.join(fig_path, "avg_cam", f"{name}", f"cluster{key}")):
                    os.mkdir(os.path.join(fig_path, "avg_cam", f"{name}", f"cluster{key}"))
                batch_feature = feature_map[name][:, activation_cluster[name][key]].cpu().detach()
                assert len(activation_cluster[name][key]) > 1
                batch_feature = batch_feature.mean(1, keepdims=True)
                batch_feature /= torch.max(batch_feature)
                for k in range(X.shape[0]):
                    origin_img = X[k].cpu().squeeze()
                    feature = batch_feature[k]
                    overlay = Cam_overlay(img=to_pil_image(origin_img), mask=to_pil_image(feature, mode='F'), max=1, alpha=0.75)
                    plt.figure()
                    plt.imshow(overlay)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_path, "avg_cam", f"{name}", f"cluster{key}", f"{k}_overlay.png"))
                    plt.close()
            else:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_param", type=str, required=True)
    parser.add_argument("--vgg_name", type=str, default="VGG16")
    #parser.add_argument("--model_param_b", type=str, default=None)
    parser.add_argument("--cluster_path", type=str, default=None)
    parser.add_argument("--distance_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--method", type=str, default="HDBSCAN", help="cluster method")
    parser.add_argument("--n_data_per_class", type=float, default=100)
    parser.add_argument("--min_sample", default=2, type=int)
    parser.add_argument("--max_ratio", default=1., type=float)
    parser.add_argument("--threshold", default=0.1, type=float)
    args = parser.parse_args()

    # load model
    model = VGG(args.vgg_name)
    checkpoint = torch.load(args.model_param)
    '''if args.model_param_b is not None:
        checkpoint_b = torch.load(args.model_param_b)'''
    model.load_state_dict(checkpoint)
    model.cuda()
    # test against mnist
    '''transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])'''
    transform = transforms.ToTensor()
    trainset = datasets.CIFAR10(root='~/data', train=True,
                                download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                               shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='~/data', train=False,
                               download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64,
                                              shuffle=False, num_workers=2)
    figure_path = '/'.join(args.model_param.split('/')[:-1])
    test_activation_cluster(model, model.state_dict(), test_loader, args.max_ratio, args.threshold, figure_path, args.vgg_name)
    #get_AVG_CAM(model, cluster, checkpoint, test_loader, figure_path)

if __name__ == "__main__":
  main()
