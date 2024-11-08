import sys
import os 
import argparse

parser = argparse.ArgumentParser() 

parser.add_argument('--dp_batchsize', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')  
parser.add_argument('--dp_norm_clip', type=float, default=1.0, metavar='M',
                    help='L2 norm clip (default: 1.0)')
parser.add_argument('--dp_noise_multiplier', type=float, default=1.0, metavar='M',
                    help='Noise multiplier (default: 1.0)')
parser.add_argument('--dp_microbatches',type=int, default=1, metavar='N',
                    help='micro batch size')
parser.add_argument('--train_size', type=int, default=10000)
parser.add_argument('--train_org', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--model_save_tag', type=str, default='0', help='a tag to be appended to the saved model path')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
from util.densenet import densenet
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.distributions import Categorical
import copy
import time

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [30, 60, 90]: 
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.


float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
random.seed(1)
# prepare test data parts
transform_train = transforms.Compose([  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataloader = datasets.CIFAR10
trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
testset = dataloader(root='./data', train=False, download=True, transform=transform_test)

X = []
Y = []
for item in trainset: 
    X.append( item[0].numpy() )
    Y.append( item[1]  )
for item in testset:
    X.append( item[0].numpy() )
    Y.append( item[1]  )
X = np.asarray(X)
Y = np.asarray(Y)

print("Loading data ..")

n_data = len(Y)
print('Total number of data points: ',n_data, flush=True)

n_train_data = len(trainset) 
n_val_data = len(testset)
n_test_data = n_val_data

if not os.path.isfile('./cifar10_shuffle.pkl'):
    all_indices = np.arange(n_data)
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open('./cifar10_shuffle.pkl','wb'))
else:
    all_indices=pickle.load(open('./cifar10_shuffle.pkl','rb'))

# Sample train, val and test sets from  the whole data
train_inputs = X[all_indices[:n_train_data]]
train_labels=Y[all_indices[:n_train_data]]
val_inputs=X[all_indices[n_train_data:]]
val_labels=Y[all_indices[n_train_data:]]
test_inputs = X[all_indices[n_train_data:]]
test_labels = Y[all_indices[n_train_data:]]

train_inputs = torch.from_numpy(train_inputs).type(torch.FloatTensor)
train_labels = torch.from_numpy(train_labels).type(torch.LongTensor)
val_inputs = torch.from_numpy(val_inputs).type(torch.FloatTensor)
val_labels = torch.from_numpy(val_labels).type(torch.LongTensor)
test_inputs = torch.from_numpy(test_inputs).type(torch.FloatTensor)
test_labels = torch.from_numpy(test_labels).type(torch.LongTensor)

print('Trainset len %d | Valset len %d Testset len %d'% (n_train_data, n_val_data, n_test_data), flush=True)


def train(train_data,labels,model,criterion,optimizer,epoch,use_cuda,num_batchs=999999,batch_size=32, uniform_reg=False):
    # switch to train mode
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    len_t = len(train_data)//batch_size
    if len(train_data)%batch_size:
        len_t += 1
    
    
    for ind in range(len_t): 

        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs  = model(inputs)
        
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)


checkpoint_dir='./checkpoints_dpsgd'
if(not os.path.exists(checkpoint_dir)):
    os.mkdir(checkpoint_dir)

num_classes = 10
BATCH_SIZE =64 
num_epochs=args.epochs
user_lr = args.lr
criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0)
model=model.cuda()
t0 = time.time()
from torch.utils.data import TensorDataset, DataLoader
def construct_new_dataloader(img_npy, y_train, batch_size=64):

    modified_train_data = []
    for i in range(len(y_train)):
       modified_train_data.append([img_npy[i], y_train[i]])
    new_train_loader = DataLoader(dataset=modified_train_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=4
                                   )
    return new_train_loader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
private_data_loader = construct_new_dataloader(train_inputs.numpy(), train_labels.numpy(), batch_size=args.dp_batchsize)
model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)
optimizer = optim.SGD(model.parameters(), lr=user_lr, momentum=0.9)
dp_delta = 1/n_train_data
privacy_engine = PrivacyEngine()
model, optimizer, private_data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=private_data_loader,
    noise_multiplier=args.dp_noise_multiplier,
    max_grad_norm=args.dp_norm_clip,
) 

best_val_acc=0
best_res = None
for epoch in range(num_epochs): 
    if True:
        lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (num_epochs + 1)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    train_loss, train_acc = train(train_inputs, train_labels, model, criterion, optimizer, epoch, 
                                    use_cuda, batch_size= BATCH_SIZE, uniform_reg=False)

    val_loss, val_acc = test(val_inputs, val_labels, model, criterion, use_cuda)

    epsilon = privacy_engine.accountant.get_epsilon(delta=dp_delta)
    res_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_acc':train_acc,
            'val_acc': val_acc,
            'optimizer': optimizer.state_dict(),
            'epsilon':epsilon,
            'time': time.time() - t0
        }
    torch.save(res_dict, os.path.join(checkpoint_dir, 'model_epoch{}_eps{:0.2f}.t7'.format(epoch+1, epsilon)))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_res = copy.deepcopy(res_dict)

    print("LR = ", optimizer.param_groups[0]['lr'])
    # Get the privacy budget expended so far
    print('Epoch %d | Train acc %.2f loss %.2f | Val acc %.2f loss %.2f | Best val acc %.2f | Epsilon %.2f'
            %(epoch+1, train_acc, train_loss, val_acc, val_loss, best_val_acc, epsilon), flush=True)
        
torch.save(best_res, os.path.join(checkpoint_dir, 'model_best_eps{:0.2f}.t7'.format(best_res['epsilon'])))
best_model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0).cuda()
best_model = ModuleValidator.fix(best_model)
ModuleValidator.validate(best_model, strict=False)
optimizer = optim.SGD(best_model.parameters(), lr=args.lr, momentum=0.9)
best_model, optimizer, data_loader = privacy_engine.make_private(
    module=best_model,
    optimizer=optimizer,
    data_loader=private_data_loader,
    noise_multiplier=args.dp_noise_multiplier,
    max_grad_norm=args.dp_norm_clip,
) 
best_model.load_state_dict(best_res['state_dict'])
best_model = best_model.cuda()
_,best_test = test(test_inputs, test_labels, best_model, criterion, use_cuda)
_,best_val = test(val_inputs, val_labels, best_model, criterion, use_cuda)
_,best_train = test(train_inputs, train_labels, best_model, criterion, use_cuda)
print('------------------Best checkpoint------------------')
print('Train acc %.4f Val acc %.4f Test acc %.4f'%(best_train, best_val, best_test), flush=True)
print('Runtime:', time.time() - t0)
print('---------------------------------------------------')