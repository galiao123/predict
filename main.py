import numpy as np
import csv
import torch
from torch.utils.data import Dataset
print('Loading data ...')

with open('data.csv') as f:
    train = np.array(list(csv.reader(f))).reshape(22, -1).astype(float).transpose()
print(train)
# with open('test.csv') as f:
#     test = np.array(list(csv.reader(f))).reshape(22, -1)
print('Size of training data: {}'.format(train.shape))
# print('Size of testing data: {}'.format(test.shape))

# ## Create Dataset
class AgriDataset(Dataset):
    def __init__(self,data,mode='train'):
        if mode is 'train':
            self.target = torch.FloatTensor(data[:,21])
        else:
            self.target = None
        self.data=data[:,:21]
    def __getitem__(self, idx):
        if self.target is not None:
            return self.data[idx], self.target[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


# Split the labeled data into a training set and a validation set, you can modify the variable `VAL_RATIO` to change the ratio of validation data.


VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train,val= train[:percent], train[percent:]
print('Size of training set: {}'.format(train.shape))
print('Size of validation set: {}'.format(val.shape))

# Create a data loader from the dataset, feel free to tweak the variable `BATCH_SIZE` here.


BATCH_SIZE = 1

from torch.utils.data import DataLoader
percent=0.1

train_set = AgriDataset(train)
val_set=AgriDataset(val)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # only shuffle the training data
val_loader=DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=False)
print(len(train_loader))
print(len(val_loader))
# Cleanup the unneeded variables to save memory.<br>
#
# **notes: if you need to use these variables later, then you may remove this block or clean up unneeded variables later<br>the data size is quite huge, so be aware of memory usage in colab**



import gc

del train
gc.collect()

# ## Create Model

# Define model architecture, you are encouraged to change and experiment with the model architecture.



import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(21, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

        self.act_fn = nn.Sigmoid()
        self.drop_out=nn.Dropout(0.3)
    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)

        return x


# ## Training


# check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Fix random seeds for reproducibility.

# In[ ]:


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Feel free to change the training parameters here.

# In[ ]:


# fix random seed for reproducibility
same_seeds(0)

# get device
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 50  # number of training epoch
learning_rate = 0.01  # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# In[ ]:


# start training

best_loss = 99.99
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    # training
    model.train()  # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.float())
        print('output:{},target:{}'.format(outputs.item(),targets.item()))
        batch_loss = criterion(outputs, targets)
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()
        train_acc+=abs(outputs.item()-targets.item())/targets.item()
    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())

                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)
                val_loss += batch_loss.item()
                val_acc += abs(outputs.item() - labels.item()) / labels.item()
            print('[{:03d}/{:03d}] Train  Acc:{:3.6f} Loss: {:3.6f} | Val Acc:{:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch,1-train_acc/len(train_loader), train_loss / len(train_loader),
               1-val_acc/len(val_loader),val_loss / len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_loss/len(val_loader) < best_loss:
                best_loss = val_loss/len(val_loader)
                torch.save(model.state_dict(), model_path)
                print('saving model with loss {:.3f}'.format(best_loss ))
    else:

        print('[{:03d}/{:03d}] Train  Loss: {:3.6f}'.format(
            epoch + 1, num_epoch,  train_loss / len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')
print("best_loss=",best_loss)
# # ## Testing
#
# # Create a testing dataset, and load model from the saved checkpoint.
#
# # In[ ]:
#
#
# # create testing dataset
# test_set = TIMITDataset(test, None)
# test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
#
# # create model and load weights from checkpoint
# model = Classifier().to(device)
# model.load_state_dict(torch.load(model_path))
#
# # Make prediction.
#
# # In[ ]:
#
#
predict = []
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(val_loader):
        inputs ,targets= data
        inputs = inputs.to(device)
        outputs = model(inputs.float())
        predict.append(outputs)
        print(targets)
#
# # Write prediction to a CSV file.
# #
# # After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.
#
# # In[ ]:


with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))






