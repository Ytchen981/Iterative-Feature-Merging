import os
import torch

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter
import time
import matplotlib.pyplot as plt

from TinyImageNet import TinyImageNet
from utils.parse_arg import parse_args, cfg
from utils.utils import set_seed, DupStdoutFileManager, print_easydict, transform_train, transform_test, \
    transform_train_norm, transform_test_norm, TinyImageNet_train_transform, TinyImageNet_test_transform

parse_args('Training model.')


def train_epoch(model, dataloader, optimizer, epoch, tfboard_writer):
    model.train()
    model.cuda()
    Acc = 0.
    total_loss = 0.
    loss_dict = dict()
    acc_dict = dict()
    for i,(X, y) in enumerate(dataloader):
        X = X.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        logit = model(X)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()
        acc = (logit.max(1)[1] == y).sum().item() / y.size(0)

        Acc += acc
        total_loss += loss.item()

        loss_dict['loss'] = loss.item()
        tfboard_writer.add_scalars('training loss', loss_dict, epoch * cfg.train.batchsize + i)

        acc_dict['accuracy'] = acc
        tfboard_writer.add_scalars(
            'training accuracy',
            acc_dict,
            epoch * cfg.train.batchsize + i
        )
    print('Train Epoch: {} \t CE: {:.4f} Acc: {:.2f}%'.format(epoch,  total_loss / (i + 1),
                                                              100 * Acc / (i + 1)))
    return total_loss / len(dataloader), 100 * Acc / len(dataloader)

def test_epoch(model, dataloader, epoch, tfboard_writer):
    model.eval()
    model.cuda()
    Acc = 0.
    total_loss = 0.
    loss_dict = dict()
    acc_dict = dict()
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()
            logit = model(X)
            loss = F.cross_entropy(logit, y)
            acc = (logit.max(1)[1] == y).sum().item() / y.size(0)

            Acc += acc
            total_loss += loss.item()

            loss_dict['testing loss'] = loss.item()
            tfboard_writer.add_scalars('testing loss', loss_dict, epoch * cfg.train.batchsize + i)

            acc_dict['testing accuracy'] = acc
            tfboard_writer.add_scalars(
                'testing accuracy',
                acc_dict,
                epoch * cfg.train.batchsize + i
            )
        print('Test Epoch: {} \t CE: {:.4f} Acc: {:.2f}%'.format(epoch,
                                                                                       total_loss / (i + 1),
                                                                                       100 * Acc / (i + 1)))
    return total_loss / len(dataloader), 100 * Acc / len(dataloader)

def main(model, trainloader, testloader, optimizer, scheduler, tfboard_writer):
    train_l, train_a, test_l, test_a = [], [], [], []
    for epoch in range(cfg.scheduler.epochs):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, epoch, tfboard_writer)
        test_loss, test_acc = test_epoch(model, testloader, epoch, tfboard_writer)
        scheduler.step()
        if (epoch+1) % 10 == 0 and epoch > 0.8 * cfg.scheduler.epochs or epoch == cfg.scheduler.epochs - 1:
            model_path = os.path.join(cfg.output_path, "params")
            if not Path(model_path).exists():
                Path(model_path).mkdir(parents=True)
            opt_path = os.path.join(model_path, f"opt_epoch{epoch}.tar")
            model_path = os.path.join(model_path, f"model_epoch{epoch}.pt")
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), opt_path)
        train_l.append(train_loss)
        train_a.append(train_acc)
        test_l.append(test_loss)
        test_a.append(test_acc)
        plot_loss(train_l, train_a, test_l, test_a)

def plot_loss(train_loss, train_acc, test_loss, test_acc):
    x = [i for i in range(len(train_loss))]
    plt.figure()
    plt.subplot(211)
    plt.plot(x, train_loss, color='red', label='train')
    plt.plot(x, test_loss, color='blue', label='test')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(212)
    plt.plot(x, train_acc, color='red', label='train')
    plt.plot(x, test_acc, color='blue', label='test')
    plt.xlabel("epoch")
    plt.ylabel("error_adv")
    plt.legend()

    plt.savefig(os.path.join(cfg.output_path, "loss_acc.png"))
    plt.close()


if __name__ == '__main__':
    set_seed(cfg.train.seed)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.output_path) / 'tensorboard' / 'training_{}'.format(now_time)))

    if cfg.dataset == "CIFAR10" or cfg.dataset == "cifar10":
        nclass=10
        if cfg.train.norm:
            trainset = datasets.CIFAR10(root=cfg.data_path, train=True, download=True, transform=transform_train_norm)
            testset = datasets.CIFAR10(root=cfg.data_path, train=False, download=True, transform=transform_test_norm)
        elif "ViT" in cfg.model:
                from utils.utils import transform_train_resize, transform_test_resize
                trainset = datasets.CIFAR10(root=cfg.data_path, train=True, download=True, transform=transform_train_resize)
                testset = datasets.CIFAR10(root=cfg.data_path, train=False, download=True, transform=transform_test_resize)
        else:
            trainset = datasets.CIFAR10(root=cfg.data_path, train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR10(root=cfg.data_path, train=False, download=True, transform=transform_test)
    elif cfg.dataset == "cifar100":
        nclass = 100
        trainset = datasets.CIFAR100(root=cfg.data_path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=cfg.data_path, train=False, download=True, transform=transform_test)
    elif cfg.dataset == "TinyImageNet":
        nclass = 200
        #"/home/chenyiting/data/tiny-imagenet-200/words.txt"
        trainset = TinyImageNet(root=cfg.data_path, split='train', transform=TinyImageNet_train_transform)
        testset = TinyImageNet(root=cfg.data_path, split='val', transform=TinyImageNet_train_transform)
    else:
        raise KeyError(f"Unknown model {cfg.dataset}")
    train_loader = DataLoader(trainset, batch_size=cfg.train.batchsize, shuffle=True, pin_memory=cfg.dataloader.pin_memory, num_workers=cfg.dataloader.num_workers)
    test_loader = DataLoader(testset, batch_size=cfg.train.batchsize, shuffle=False, pin_memory=cfg.dataloader.pin_memory, num_workers=cfg.dataloader.num_workers)



    if cfg.model == "vgg11":
        from model.vgg import VGG
        model = VGG('VGG11', nclass)
    elif cfg.model == "vgg13":
        from model.vgg import VGG
        model = VGG('VGG13', nclass)
    elif cfg.model == "vgg16":
        from model.vgg import VGG
        model = VGG('VGG16', nclass)
    elif cfg.model == "vgg19":
        from model.vgg import VGG
        model = VGG('VGG19', nclass)
    elif "VGG" in cfg.model:
        from model.vgg import VGG
        model = VGG(cfg.model, nclass)
    elif cfg.model == "ResNet18":
        from model.resnet import ResNet18
        model = ResNet18()
    elif cfg.model == "ResNet34":
        from model.resnet import ResNet34
        model = ResNet34()
    elif cfg.model == "ResNet50":
        from model.resnet import ResNet50
        model = ResNet50()
    else:
        raise KeyError(f"Unknown model {cfg.model}")

    optimizer = optim.SGD(params=model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum,
                           weight_decay=cfg.train.weight_decay)


    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler.milestones,
                                               gamma=cfg.scheduler.lr_decay)


    with DupStdoutFileManager(str(Path(cfg.output_path) / (now_time + '.log'))) as _:
        print_easydict(cfg)
        main(model, train_loader, test_loader, optimizer, scheduler, tfboardwriter)
