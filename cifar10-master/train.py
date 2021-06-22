from __future__ import print_function
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger

import models
def data_constrcting(train_data,test_data,args):
    #  获得训练集中用于作为验证集的数据索引
    batch_size = args.batch_size
    valid_size = args.valid_size
    num_train = len(train_data)
    indices = list(range(num_train))
    if args.shuffle:
        np.random.shuffle(indices)
    split = int(valid_size * num_train)
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
        sampler = train_sampler, num_workers = 0)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, 
        sampler = valid_sampler, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, 
        num_workers = 0) 
    return train_loader, valid_loader, test_loader

def cutout_data(inputs, targets, ratio=0.2, use_cuda=True):
    h = inputs.size(-2)
    w = inputs.size(-1)
    mask = np.ones((h, w), np.float32)
    for n in range(int(ratio*h*w)):
        y = np.random.randint(h)
        x = np.random.randint(w)
        mask[y,x] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(inputs)
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
        mask = mask.cuda()
    inputs = inputs * mask
    return inputs, targets

def train(epoch):
    print('\nEpoch: %d' % epoch)
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        model.zero_grad()
        pred = model(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)
    return accuracy, xentropy_loss_avg

def train_cutout(epoch):
    print('\nEpoch: %d' % epoch)
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()
        images, labels = cutout_data(images, labels)
        model.zero_grad()
        pred = model(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)
    return accuracy, xentropy_loss_avg

def train_mixup(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, args.cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        model.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(
            xentropy='%.3f' % (train_loss / (batch_idx + 1)),
            acc='%.3f' % (100. * correct / total))
    return (correct / total).item(), train_loss


def train_cutmix(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        r = np.random.rand(1)
        if args.alpha > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.alpha, args.alpha)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            output = model(inputs)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            # compute output
            output = model(inputs)
            loss = criterion(output, targets)

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(
            xentropy='%.3f' % (train_loss / (batch_idx + 1)),
            acc='%.3f' % (100. * correct / total))
    return (correct / total).item(), train_loss


def test():
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    test_loss = 0
    # y_true, y_pred = [], []
    for images, labels in valid_loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)
        test_loss += criterion(pred,labels).item()
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    #     y_true.append(labels)
    #     y_pred.append(pred)
    test_acc = correct / total
    test_loss = test_loss / total
    model.train()
    return test_acc, test_loss

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--method', default='baseline')
parser.add_argument('--shuffle', type=bool, default=True,
                    help='whether to shuffle the trainset')
parser.add_argument('--dataset', '-d', default='cifar10',
                    help='cifar10, cifar100')
parser.add_argument('--model', '-a', default='lenet',
                    help='densenet, googlenet, lenet, resnet18, vgg')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--valid_size', type=int, default=0.2,
                    help='proportion of validation')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16, help='length of the holes')
parser.add_argument('--alpha', default=0.2, type=float, help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--cutmix_prob', default=0.1, type=float, help='cutmix probability')
args = parser.parse_args(args=[])
args.method = 'cutout'
# args = parser.parse_args()

if args.method == 'cutout':
    from util.cutout import Cutout
    train = train_cutout
elif args.method == 'mixup':
    from util.mixup import mixup_data, mixup_criterion
    train = train_mixup
elif args.method == 'cutmix':
    from util.cutmix import rand_bbox, mixup_criterion
    train = train_cutmix
elif args.method == 'baseline':
    train = train
else:
    raise Exception('unknown method: {}'.format(args.method))
    
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.method == 'cutout':
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

num_classes = 10
train_dataset = datasets.CIFAR10(root='./data/',
                                 train=True,
                                 transform=train_transform,
                                 download=True)

test_dataset = datasets.CIFAR10(root='./data/',
                                train=False,
                                transform=test_transform,
                                download=True)

# Data Loader (Input Pipeline)
train_loader, valid_loader, test_loader = data_constrcting(train_dataset,test_dataset,args)

model = models.lenet.LeNet()

model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                            momentum=0.9, nesterov=True, weight_decay=5e-4)

scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

test_id = args.dataset + '_' + args.model + '_' + args.method
if not args.data_augmentation:
    test_id += '_noaugment'
filename = './runs/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'valid_acc', 'train_loss', 'valid_loss']
                       , filename=filename)
best_scores = 100000000
for epoch in range(1, args.epochs + 1):
    train_acc, train_loss = train(epoch)
    val_acc, val_loss = test()
    tqdm.write('test_acc: %.3f' % val_acc)
    scheduler.step()
    row = {'epoch': str(epoch), 'train_acc': str(train_acc), 'valid_acc': str(val_acc), 
           'train_loss': str(train_loss), 'valid_loss': str(val_loss)}
    csv_logger.writerow(row)
    if val_loss < best_scores:
        torch.save(model.state_dict(), './check/' + test_id + '.pt')
csv_logger.close()

def acc_loss(method):
    model_file = './check/'+'cifar10_lenet_'+method+'_noaugment.pt'
    model_ = models.lenet.LeNet()
    model_.load_state_dict(torch.load(model_file))
    model_.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    test_loss = 0
    y_true, y_pred = [], []
    for images, labels in test_loader:
        with torch.no_grad():
            pred = model_(images)
        test_loss += criterion(pred,labels).item()
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        y_true.append(labels)
        y_pred.append(pred)
    test_acc = correct / total
    test_loss = test_loss / total
    return y_true, y_pred, test_acc, test_loss
for method in ['baseline', 'mixup', 'cutout', 'cutmix']:
    fig = plt.figure(figsize=(8,6))
    y_true, y_pred, test_acc, test_loss = acc_loss(method)
    print(test_acc,test_loss)
    metrics.confusion_matrix(np.array(torch.cat(y_true).cpu()),np.array(torch.cat(y_pred).cpu()))
    sns.heatmap(metrics.confusion_matrix(np.array(torch.cat(y_true).cpu()),np.array(torch.cat(y_pred).cpu())))
    title = "LeNet(" + method + ")"
    plt.title(title)
    plt.savefig(fname=title)
    plt.show()