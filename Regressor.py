#!/usr/bin/env python
# coding: utf-8

# In[1]:


from comet_ml import Experiment
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.autograd.variable import Variable
import sys
import random
import time
import h5py

from models.getting_high_constrainer import energyRegressor
#from models.Simple_constrainer import Regressor
#from models.constrainer_5_layer_FC_dropout_BN import Regressor

from HDF5Dataset import HDF5Dataset

import matplotlib.pyplot as plt
import matplotlib as mpl


# In[2]:


thresh = 0.0      ## transformation: kein cut auf photonen

# Transformation for lower cut on spectrum
def tf_Photons_thresh(x):
        x[x < thresh] = 0.0
        return x


# In[47]:


# Constants  ## hyperparameters   
###random Seed = zufallsfunktion werden festgelegt - reproduzierbarkeit
manualSeed = 2517
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Number of workers for dataloader - parallele instanzen dataloader
workers = 20
# batch size
batch_size = 64 #32

# Number of training epochs
num_epochs = 500 #50 #100 #200   ### zum testen 2, später 200 sinnvoll

# learning rate for constrianer optimizer  ###erstmal so anfangen, default(1e-3), rumprobieren
lr_c = 1e-3
#### nach dem training loss curve / comet - training und test loss vs epoche - learning rate optimiert = loss schnell auf sehr gringe zahl


# hyperparameter for the Adam optimizer
### default 0,9 - erstmal bei 0,5 belassen, wenn alles läuft ausprobieren - peter grund?
beta1 = 0.9
# Number of GPUs available- if 0 then CPU mode
ngpu = 1
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# In[48]:


# Set path and create function for saving and loading checkpoints
saving_path = '/beegfs/desy/user/werthern/Regression/For_Nana/Output/TestK/TestK_{}.pt'   




### in andere funktion - importieren
def save(net, optim, epoch, train_loss, validation_loss, path_to_save):
    torch.save({
        'Angular Constrainer': net.state_dict(),
        'Angular Constrainer optimizer': optim.state_dict(),
        'epoch': epoch,
        'Training loss': train_loss,
        'Validation loss': validation_loss
    },
        path_to_save.format(epoch))


# In[49]:


# Instantiate model/loss/optimizer classes and initialize functions     
net = energyRegressor().to(device) #netC_Ang

# Loss class- L1
criterion = nn.L1Loss() #criterion_c_Ang

# Randomly initialize the weights
#net.apply(weights_init)

# Use Adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr_c, betas=(beta1, 0.999))

epoch_checkpoint = 0 
train_losses = np.array([])
validation_losses = np.array([])
Avg_accuracy = np.array([])
Avg_error = np.array([])

net.train()   ###netzwerk in training modus - weights variabel - später wiedel .eval()

print('Loading data...')


# In[50]:


loader_params = {'shuffle': True, 'num_workers': 1}     # 10-20% validationset - 80/20
tf = lambda x:(x)

input_path = '/beegfs/desy/user/diefenbs/shower_data/gamma-fullG-950kCorrected.hdf5'
train_size = 150000   # size of dataset ###950.000
batch_size = 128

dataset = HDF5Dataset(input_path, transform=tf, train_size=train_size)
dataset_train, dataset_val = torch.utils.data.random_split(dataset, 
                                                               [int(0.80*train_size), 
                                                                train_size - int(0.80*train_size)])

data_loader = torch.utils.data.DataLoader(dataset, **loader_params)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, **loader_params)   #25000

test_loader = torch.utils.data.DataLoader(dataset_val, batch_size = batch_size, **loader_params)      #25000,#500000

### testset - 5% training size - erst wenns läuft
#validationset umbenennen (nach jeder epoche=val)


#epcohe mit geringstem validation loss für berechnung

print('Loading done')


# In[51]:


# create CometML experiment to see how training is progressing in real time
### jedes mal projectname umbenennen - dokumentation nachlesen
experiment = Experiment(
    api_key="fOyFCRFHFWbX1PSybMjU14oi0",
    project_name="ba-regression",
    workspace="nanamarieluisa",
)


experiment.add_tag('ba_regression_calorimeterdata_testK')
experiment.log_parameter('Manual Seed', manualSeed)
experiment.log_parameter('Batch_size', batch_size)
experiment.log_parameter('Number of Epochs', num_epochs)
experiment.log_parameter('Learning rate for Angular constrainer optimizer', lr_c)


experiment.set_model_graph(str(net), overwrite=False) ###??? dokumentation - netzwerk selbst hochgeladen?


# In[53]:


print('Begin training')
train_loss_avg = []
val_loss_avg = []
train_loss_avg_end = np.array([])
val_losses = np.array([])


for epoch in range(num_epochs):
    epoch += epoch_checkpoint + 1
    
    epoch_time = time.time()
    net.train()
    for batch_idx, (data, energy) in enumerate(train_loader):    
        batch_size = len(data)
        true_shower = Variable(data.unsqueeze(1)).float().to(device).view(batch_size, 1, 30, 30, 30)
        true_energy = Variable(energy.unsqueeze(1)).float().to(device).view(batch_size, 1, 1, 1, 1)         
        
        # Clear gradients- apply to model rather than optimizer
        net.zero_grad()
        
        # Forward pass showers through the constrainer
        output = net(true_shower)
        
        # Calculate constrainers' loss on output
        train_loss = criterion(output, true_energy.view(batch_size, 1))
        
        # Backpropagate the constrainers' gradients
        train_loss.backward()
        
        # Update step on constrainer
        optimizer.step()
        
        # Output training stats
        train_losses = np.append(train_losses, train_loss.item())
        train_loss_end = train_loss.item()
        
        if batch_idx % 1 ==0:
            print('[%d/%d] [%d/%d], (Training Loss_C: %.4f)'%(epoch, num_epochs, batch_idx, len(train_loader), train_loss.item()))
    
    train_losses_mean = np.mean(train_losses)
    train_loss_avg.append(train_losses_mean)
    train_loss_avg_end = np.append(train_loss_avg_end,train_loss_end)
    
    # Now test model performance on unseen data
    summed_accuracy = []
    summed_err = []
    size = []
    size_err = []
    with torch.no_grad():
        net.eval()
        for batch_idx, (data, energy) in enumerate(test_loader):
            batch_size = len(data)
            true_shower_val = Variable(data.unsqueeze(1)).float().to(device).view(batch_size, 1, 30, 30, 30) 
            true_energy_val = Variable(energy.unsqueeze(1)).float().to(device).view(batch_size, 1, 1, 1, 1)
            
            # Forward pass only to get output
            val_outputs = net(true_shower_val)
            
            # Calculate validation loss
            val_loss = criterion(val_outputs, true_energy_val.view(batch_size, 1))
            val_losses = np.append(val_losses, val_loss.item())
            
            if batch_idx % 1 ==0:
                print('[%d/%d] [%d/%d], (Test Loss_C: %.4f)'%(epoch, num_epochs, batch_idx, len(test_loader), val_loss.item()))
            
            # Calculate error (absolute percentage error of predictions given labels)
            percentage_error = abs((val_outputs - true_energy_val.view(batch_size, 1))/true_energy_val.view(batch_size, 1))*100
        
            # Calculate accuracy
            accuracy = 100 - percentage_error
            
            summed_accuracy.append(torch.sum(accuracy))        
            size.append(len(accuracy))
            
            summed_err.append(torch.sum(percentage_error))
            size_err.append(len(percentage_error))
        
        val_losses_mean = np.mean(val_losses)
        #val_loss_mean = sum(val_losses)/len(val_losses)
        val_loss_avg.append(val_losses_mean)
        
        # Summed Accuracy
        total_summed_accuracy = sum(summed_accuracy)
        total_size = sum(size)
        avg = total_summed_accuracy/total_size
        
        # Summed Error
        total_summed_err = sum(summed_err)
        total_size_err = sum(size_err)
        avg_err = total_summed_err/total_size_err
        
        # Über gesamtes Testset
        Avg_accuracy = np.append(Avg_accuracy, avg.item())
        Avg_error = np.append(Avg_error, avg_err.item())
       
        
        ###############################
        # Log  to COMET ML
        ###############################

        experiment.log_metric('Training Loss', train_losses_mean, epoch=epoch)   # nicht 1:1 vergleichbar, da netzwerk sich wären Epoche verändert
        experiment.log_metric('Training Loss Epoch End', train_loss_end, epoch=epoch)
        experiment.log_metric('Validation Loss', val_losses_mean, epoch=epoch)
        experiment.log_metric('Accuracy', Avg_accuracy, epoch=epoch)
        experiment.log_metric('Percentage error', Avg_error, epoch=epoch)
        
        #print('Loss Training Epoch Mean: ',train_losses_mean)
        #print('Loss Training Epoch End :',train_loss_avg_end)
        #print('Loss Validation: ',val_losses_mean)
        
        #np.savetxt('train_losses_mean.txt', train_losses_mean)
        #np.savetxt('val_loss_mean.txt', val_losses_mean)
        #np.savetxt('train_loss_avg_end.txt', train_loss_avg_end)

    save(net=net, optim=optimizer, epoch=epoch, train_loss=train_loss_avg, validation_loss=val_loss_avg, path_to_save=saving_path)
    
    print("saving epoch done")


# In[54]:


print(train_loss_avg)
print(val_loss_avg)
print(train_loss_avg_end)


# In[55]:


##plot single energies - echte energie vs. energie loss
## datenset:plotting script, single energy datanset, energie (10.000-50.000 samples für verschiedene GeV sets) durch das netzwerk mit dem geringsten validation loss
# netzwerk laden, datenset durchschicken


# In[56]:


# Plot loss when training is complete
plt.figure(figsize=(5,5))
plt.title("Loss during training")
plt.plot(train_loss_avg, label="Loss")
plt.plot(val_loss_avg, label="Validation loss")
plt.plot(train_loss_avg_end, label="Loss Epoch End")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.tight_layout()
plt.legend()
plt.show()
experiment.log_figure(figure=plt, figure_name="Loss during training")
plt.savefig('/beegfs/desy/user/werthern/Regression/For_Nana/Output/plots/loss_testK.png')


# In[57]:


# Plot Accuracy when training is complete
plt.figure(figsize=(5,5))
plt.title("Accuracy during training")
plt.plot(Avg_accuracy,label="Accuracy")
plt.plot(Avg_error,label="Error")
plt.xlabel("Epochs")
plt.ylabel("Percentage accuracy/error")
plt.legend()
plt.show()
experiment.log_figure(figure=plt, figure_name="Accuracy during training")        
plt.savefig('/beegfs/desy/user/werthern/Regression/For_Nana/Output/plots/acuracy_testK.png')       


# In[ ]:




