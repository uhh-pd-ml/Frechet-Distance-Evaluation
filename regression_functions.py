#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time
import pickle
import h5py
#sys.path.append('/home/buhmae/1_GenerateCalice/GenerateCalice')
sys.path.append('/beegfs/desy/user/werthern/CalorimeterData/for_Nana/GenerateCalice')
#sys.path.append('/home/buhmae/1_UnderstandingVAE')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as clr
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import importlib
#from ipywidgets import interact, fixed
#import ipywidgets as widgets
get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib notebook

# custom libraries
#import VAE.VAE_Model.VAE_Models as VAE_Models
import VAE_ML.ML_models.models as modelsVAE
import VAE_ML.Generate_Plots_F_Erik as Generate_Plots
#import VAE_ML.Generate_Plots_F as Generate_Plots
#import VAE.VAE_Model.VAE_functions as VAE_functions
import MyFunctionsErik as myF
#importlib.reload(VAE_Models)
importlib.reload(Generate_Plots)
importlib.reload(modelsVAE)
#importlib.reload(VAE_functions)
importlib.reload(myF)

import torchvision.models as models
import torch.nn.functional as F
from scipy import linalg

import torch
import torch.nn as nn

from models.getting_high_constrainer_frechet import energyRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[2]:


def generateData (weights, directory, f_name, numEvents, real_energy, use_PP):
    
    
    latent_small = 24   # 12 or 24 or 512
    latent = 512
    #use_PP = False   # use Post Processing
    global_thresh = 0.1    # MIP cut

    save_locations = {
        "Full" : '/beegfs/desy/user/eren/test_area/ILDConfig/StandardConfig/production/50k.hdf5',
        "Single" : '/beegfs/desy/user/eren/WassersteinGAN/data/singleEnergies-corr.hdf5',
        }
    
    
    # load model
    args = {"E_cond" : True, 
              "latent": latent}
    
    # lade checkpoint:
    checkpoint = torch.load(weights, map_location=torch.device(device))

    # small model for (variable) lat (default: 24) / kld=0.1 & 0.05
    model = modelsVAE.BiBAE_F_3D_LayerNorm_SmallLatent(args, device=device, z_rand=latent-latent_small, z_enc=latent_small).to(device)   
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])

    if use_PP == True:
        model_P = modelsVAE.PostProcess_Size1Conv_EcondV2(bias=True, out_funct='none').to(device)   # default from paper
        model_P = nn.DataParallel(model_P)
        model_P.load_state_dict(checkpoint['model_P_state_dict'])
        print('using Post Processing')
    else:
        print('not using Post Processing')
    
    #print(model)
    #if use_PP:
    #    print(model_P)
        
    
    # generate same amount of fake images & latent space as real events
    batchsize = 100

    if use_PP == True:
        # full energy range
        fake_images, z_fake = myF.getFakeImages_3d_PP_all(model, model_P, real_energy, numEvents, batchsize=batchsize, latent_dim=latent, device=device, thresh=global_thresh)
    else:
        fake_images, z_fake = myF.getFakeImages_3d_noPP_all(model, real_energy, numEvents, batchsize=batchsize, latent_dim=latent, device=device, thresh=global_thresh)
    
    return fake_images

    # save h5 files
    h5f = directory+f_name
    with h5py.File(h5f, 'w') as hf:  
        hf.create_dataset('fake_images', data=fake_images)
    
    print('saving done')




# In[3]:


## Resize Images

def resizeRealImages(data_real):
    # Create Real Image List
    real_images = np.array(data_real['real_images'])
    
    # Transform real images
    real_images_2d = np.sum(real_images,3)
    real_images_2d.shape
    
    real_images_3d = np.expand_dims(real_images_2d, axis = 1)
    real_images = np.repeat(real_images_3d, repeats=3, axis=1)
    real_images.shape
    
    img = torch.tensor(real_images)
    
    # resize real images
    import torch.nn.functional as F
    real_img = F.interpolate(img, size=64)
    return real_img


def resizeSingleEnergiesReal(data_real):
    real_images = np.array(data_real['30x30']['layers'])
        # Transform real images
    real_images_2d = np.sum(real_images,3)
    real_images_2d.shape
    
    real_images_3d = np.expand_dims(real_images_2d, axis = 1)
    real_images = np.repeat(real_images_3d, repeats=3, axis=1)
    real_images.shape
    
    img = torch.tensor(real_images)
    
    # resize real images
    import torch.nn.functional as F
    real_img = F.interpolate(img, size=64)
    return real_img


def resizeRealImagesLayer(data_real):
    # Create Real Image List
    real_images = np.array(data_real['real_images'])
    
    # Transform real images
    real_images_expand = np.expand_dims(real_images, axis = 1)
    real_images = np.repeat(real_images_expand, repeats=3, axis=1)
    
    real_images = np.transpose(real_images, (4,0,1,2,3))
    img = torch.tensor(real_images)
    
    return img


def resizeFakeImagesLayer(fake_images):
    
    # Transform fake images
    fake_images_expand = np.expand_dims(fake_images, axis = 1)
    fake_images = np.repeat(fake_images_expand, repeats=3, axis=1)

    fake_images = np.transpose(fake_images, (4,0,1,2,3))
    
    fimg = torch.tensor(fake_images)
    return fimg



## Resize Energy


def resizeRealEnergy(data_real):
    # Create Real Image List
    real_energy = np.array(data_real['real_energy'])
    
    # Transform real images
    #real_energy_2d = np.sum(real_energy,3)
    #real_energy_2d.shape
    
    #real_energy_3d = np.expand_dims(real_energy_2d, axis = 1)
    #real_energy = np.repeat(real_energy_3d, repeats=3, axis=1)
    #real_energy.shape
    
    #img = torch.tensor(real_energy)
    
    # resize real images
    #import torch.nn.functional as F
    #real_energy = F.interpolate(img, size=64)
    return real_energy

def resizeRealEnergyRegression(data_real):
    # Create Real Image List
    real_energy = np.array(data_real['30x30']['energy'])
    return real_energy

def resizeSingleEnergiesEnergy(data_real):
    # Create Real Image List
    real_energy = np.array(data_real['30x30']['energy'])
    return real_energy


# In[5]:


def resizeFakeImages(fake_images):
    #fake_images = np.array(fake_data['fake_images'])
  
    # Transform fake images
    fake_images_2d = np.sum(fake_images,3)
    fake_images_2d.shape
    
    fake_images_3d = np.expand_dims(fake_images_2d, axis = 1)
    fake_images = np.repeat(fake_images_3d, repeats=3, axis=1)
    fake_images.shape
    
    fimg = torch.tensor(fake_images)
    
    # resize fake images
    import torch.nn.functional as F
    fake_img = F.interpolate(fimg, size=64)
    return fake_img


# In[6]:


def calculate_activation_statistics(images,model,batch_size=200, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# In[7]:


def calculate_act(images,model,batch_size=200, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()    #tensor = torch.from_numpy(array)
    else:
        batch=images
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    return act


# In[8]:


def calculate_fretchet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape,         'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape,         'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


# In[9]:


def calculate_fretchet(images_real,images_fake,model):
     mu_1,std_1=calculate_activation_statistics(real_images,model,cuda=True)
     mu_2,std_2=calculate_activation_statistics(fake_images,model,cuda=True)
    
     """get freched distance"""
     fid_value = calculate_fretchet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value


# In[10]:


def calculateFrechet(real_img, fake_img, batch_size):
    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    class InceptionV3(nn.Module):
        """Pretrained InceptionV3 network returning feature maps"""

        # Index of default block of inception to return,
        # corresponds to output of final average pooling
        DEFAULT_BLOCK_INDEX = 3

        # Maps feature dimensionality to their output blocks indices
        BLOCK_INDEX_BY_DIM = {
            64: 0,   # First max pooling features
            192: 1,  # Second max pooling featurs
            768: 2,  # Pre-aux classifier features
            2048: 3  # Final average pooling features
        }

        def __init__(self,
                     output_blocks=[DEFAULT_BLOCK_INDEX],
                     resize_input=True,
                     normalize_input=True,
                     requires_grad=False):

            super(InceptionV3, self).__init__()

            self.resize_input = resize_input
            self.normalize_input = normalize_input
            self.output_blocks = sorted(output_blocks)
            self.last_needed_block = max(output_blocks)

            assert self.last_needed_block <= 3,                 'Last possible output block index is 3'

            self.blocks = nn.ModuleList()


            inception = models.inception_v3(pretrained=True)

            # Block 0: input to maxpool1
            block0 = [
                inception.Conv2d_1a_3x3,
                inception.Conv2d_2a_3x3,
                inception.Conv2d_2b_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block0))

            # Block 1: maxpool1 to maxpool2
            if self.last_needed_block >= 1:
                block1 = [
                    inception.Conv2d_3b_1x1,
                    inception.Conv2d_4a_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2)
                ]
                self.blocks.append(nn.Sequential(*block1))

            # Block 2: maxpool2 to aux classifier
            if self.last_needed_block >= 2:
                block2 = [
                    inception.Mixed_5b,
                    inception.Mixed_5c,
                    inception.Mixed_5d,
                    inception.Mixed_6a,
                    inception.Mixed_6b,
                    inception.Mixed_6c,
                    inception.Mixed_6d,
                    inception.Mixed_6e,
                ]
                self.blocks.append(nn.Sequential(*block2))

            # Block 3: aux classifier to final avgpool
            if self.last_needed_block >= 3:
                block3 = [
                    inception.Mixed_7a,
                    inception.Mixed_7b,
                    inception.Mixed_7c,
                    nn.AdaptiveAvgPool2d(output_size=(1, 1))
                ]
                self.blocks.append(nn.Sequential(*block3))

            for param in self.parameters():
                param.requires_grad = requires_grad

        def forward(self, inp):
            """Get Inception feature maps
            Parameters
            ----------
            inp : torch.autograd.Variable
                Input tensor of shape Bx3xHxW. Values are expected to be in
                range (0, 1)
            Returns
            -------
            List of torch.autograd.Variable, corresponding to the selected output
            block, sorted ascending by index
            """
            outp = []
            x = inp

            if self.resize_input:
                x = F.interpolate(x,
                                  size=(299, 299),
                                  mode='bilinear',
                                  align_corners=False)

            if self.normalize_input:
                x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

            for idx, block in enumerate(self.blocks):
                x = block(x)
                if idx in self.output_blocks:
                    outp.append(x)

                if idx == self.last_needed_block:
                    break

            return outp

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model=model.cuda()
    
    
    
    
    
    # calculate frechet
    
    iters = 0
    print("checking real image shape")
    print(real_img.shape)
    for i in range(0, len(real_img), batch_size):
         data_batch_real = real_img[i:i+200]
         data_batch_real = data_batch_real.float()

         #act
         act_real_list = []
         act_real = calculate_act(data_batch_real,model,batch_size=200, dims=2048, cuda=True)
         act_real_list.append(act_real)

         iters += 1 

    print("finished calculating act_real")

    iters = 0
    print("checking fake image shape")
    print(fake_img.shape)
    for i in range(0, len(fake_img), batch_size):    
         data_batch_fake = fake_img[i:i+200]
         data_batch_fake = data_batch_fake.float()

         #act
         act_fake_list = []
         act_fake = calculate_act(data_batch_fake,model,batch_size=200, dims=2048, cuda=True)
         act_fake_list.append(act_fake)

         iters += 1 

    print("finished calculating act_fake")

    # Calculate Fretchet distance
    act_real_all = np.vstack(act_real_list)
    act_fake_all = np.vstack(act_fake_list)

    mu_real = np.mean(act_real_all, axis=0)             #vorher act_real und act_fake
    sigma_real = np.cov(act_real_all, rowvar=False)
    mu_fake = np.mean(act_fake_all, axis=0)
    sigma_fake = np.cov(act_fake_all, rowvar=False)


    fid_value = calculate_fretchet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    print("finished calculating fid value")
    return fid_value

