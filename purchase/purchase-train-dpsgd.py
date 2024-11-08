import sys 
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import time
from tqdm import tqdm

import argparse
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
parser = argparse.ArgumentParser()  
parser.add_argument('--dp_batchsize', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')  
parser.add_argument('--dp_norm_clip', type=float, default=1.0, metavar='M',
                    help='L2 norm clip (default: 1.0)')
parser.add_argument('--dp_noise_multiplier', type=float, default=1.0, metavar='M',
                    help='Noise multiplier (default: 1.0)')
parser.add_argument('--dp_microbatches',type=int, default=1, metavar='N',
                    help='micro batch size')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()
 
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

print("Loading data...", flush=True)
data_set= np.load('./purchase.npy')
X = data_set[:,1:].astype(np.float64)
Y = (data_set[:,0]).astype(np.int32)-1

n_data = len(Y)
print('Total number of data points: ',n_data, flush=True)

n_train_data = int(n_data * 0.8) 
n_val_data = int(n_data * 0.2)
n_test_data = int(n_data * 0.2) 

if not os.path.isfile('./purchase_shuffle.pkl'):
    all_indices = np.arange(n_data)
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open('./purchase_shuffle.pkl','wb'))
else:
    all_indices=pickle.load(open('./purchase_shuffle.pkl','rb'))

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

class PurchaseClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(PurchaseClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(600,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,num_classes)
        
    def forward(self,inp):
        
        outputs=[]
        x=inp
        module_list =list(self.features.modules())[1:]
        for l in module_list:
            
            x = l(x)
            outputs.append(x)
        
        y = x.view(inp.size(0), -1)
        o = self.classifier(y)
        
        return o, outputs[-1].view(inp.size(0), -1), outputs[-4].view(inp.size(0), -1)


def train(data_loader,model,criterion,optimizer,epoch,use_cuda,num_batchs=999999,batch_size=32):
    # switch to train mode
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
     
    for (inputs, targets) in data_loader: 
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _, _  = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGdistil_epochsD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 

    return (losses.avg, top1.avg)

n_classes=100
criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
model=PurchaseClassifier()
model=model.cuda()
best_val_acc=0
from torch.utils.data import TensorDataset, DataLoader
def construct_new_dataloader(img_npy, y_train, batch_size=64):

    modified_train_data = []
    for i in range(len(y_train)):
       modified_train_data.append([img_npy[i], y_train[i]])
    new_train_loader = DataLoader(dataset=modified_train_data,
                                   batch_size=batch_size,
                                   shuffle=False
                                   )
    return new_train_loader


t0 = time.time()
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import copy
private_data_loader = construct_new_dataloader(train_inputs.numpy(), train_labels.numpy(), batch_size=args.dp_batchsize)
model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
dp_delta = 1/n_train_data
privacy_engine = PrivacyEngine()
model, optimizer, private_data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=private_data_loader,
    noise_multiplier=args.dp_noise_multiplier,
    max_grad_norm=args.dp_norm_clip,
) 

checkpoint_dir='./checkpoints_dpsgd'
if(not os.path.exists(checkpoint_dir)):
    os.mkdir(checkpoint_dir)
best_train_acc = 0.
best_res = None
for epoch in tqdm(range(args.epochs), desc="Training epochs"):
    if True:
        lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


    train_loss, train_acc = train(private_data_loader, model, criterion, optimizer, epoch, use_cuda, batch_size= args.dp_microbatches)

    train_loss,train_acc = test(train_inputs, train_labels, model, criterion, use_cuda, device=device)        
    val_loss, val_acc = test(val_inputs, val_labels, model, criterion, use_cuda, device=device)
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

best_model=PurchaseClassifier().cuda()
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






