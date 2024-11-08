'''
Code adapted from https://github.com/vrt1shjwlkr/AAAI21-MIA-Defense
'''
import sys
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
from torch.distributions import Categorical
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--wd', type=float, default=1e-3)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

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


def train(train_data,labels,model,criterion,optimizer,epoch,use_cuda,num_batchs=999999,batch_size=32, uniform_reg=False):
    # switch to train mode
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    n_batches = len(train_data)//batch_size
    if len(train_data)%batch_size:
        n_batches+= 1
    
    for ind in range(n_batches):
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs,_,_ = model(inputs)
        
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)


checkpoint_dir='./checkpoints_reg'
if(not os.path.exists(checkpoint_dir)):
    os.mkdir(checkpoint_dir)

ALPHA = 1
BATCH_SIZE = 32
num_epochs = 200
train_lr =0.0005
weight_decay=args.wd

criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
model=PurchaseClassifier()
model=model.cuda()
optimizer=optim.Adam(model.parameters(), lr=train_lr, weight_decay=weight_decay)
best_val_acc=0
t0 = time.time()
for epoch in tqdm(range(num_epochs), desc="Training epochs"):
    train_loss, train_acc = train(train_inputs, train_labels, model, criterion, optimizer, epoch, 
                                    use_cuda, batch_size= BATCH_SIZE, uniform_reg=False)

    val_loss, val_acc = test(val_inputs, val_labels, model, criterion, use_cuda)

    res_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_acc':train_acc,
            'val_acc': val_acc,
            'optimizer': optimizer.state_dict(),
            'time':time.time() - t0
        }
    torch.save(res_dict, os.path.join(checkpoint_dir, 'model_epoch{}_wd{:0.4f}.t7'.format(epoch+1, weight_decay)))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(res_dict, os.path.join(checkpoint_dir, 'model_best_wd{:0.4f}.t7'.format(weight_decay)))

    print('Epoch %d | Train acc %.2f loss %.2f | Val acc %.2f loss %.2f | Best val acc %.2f'
            %(epoch+1, train_acc, train_loss, val_acc, val_loss, best_val_acc, ), flush=True)


best_model=PurchaseClassifier().cuda()
checkpoint = torch.load(os.path.join(checkpoint_dir, 'model_best_wd{:0.4f}.t7'.format(weight_decay)))
best_model.load_state_dict(checkpoint['state_dict'])
_,best_test = test(test_inputs, test_labels, best_model, criterion, use_cuda)
_,best_val = test(val_inputs, val_labels, best_model, criterion, use_cuda)
_,best_train = test(train_inputs, train_labels, best_model, criterion, use_cuda)
print('------------------Best checkpoint------------------')
print('Train acc %.4f Val acc %.4f Test acc %.4f'%(best_train, best_val, best_test), flush=True)
print('Runtime:', time.time() - t0)
print('---------------------------------------------------')