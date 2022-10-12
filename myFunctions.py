# imports
import sys
sys.path.append('/beegfs/desy/user/buhmae/for_Nana/GenerateCalice')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as clr
import scipy.spatial.distance as dist
from scipy import stats

from VAE_ML.data_utils.data_loader import HDF5Dataset

# custom libraries
import VAE_ML.ML_models.models as VAE_Models
#import VAE.VAE_Model.VAE_functions as VAE_functions
import VAE_ML.data_utils.data_loader as loader
import VAE_ML.Generate_Plots_F_Erik as Generate_Plots

import torch

#import pickle
import dill as pickle 

###########################################################3

### GENERAL FUNCTIONS ###
tf_logscale = lambda x:(np.log((np.sum(x, axis=0)+1.0)))
tf_logscale_rev = lambda x:(torch.exp(x)-1.0)
tf_logscale_rev_np = lambda x:(np.exp(x)-1.0)
E_true_trans = lambda x:(x*1e-2)


# KLD divergence
def KLD_nats(mu, std):
    B = mu.shape[0]

    logvar = 2 * np.log(std)
    KLD = (-0.5 * np.sum(1 + logvar - np.power(mu, 2) - np.exp(logvar)))/(B)

    return KLD

def KLD_bits(mu, std):
    return KLD_nats(mu,std) / np.log(2)

def number_KLD_bits_thresh(mu, std, thresh = 1.0):
    tmp_list = []
    for i in range(mu.shape[1]):
        KLD = KLD_bits(mu[:,i],std[:,i])
        tmp_list.append(KLD)
    return np.sum(np.array(tmp_list)>=thresh)


# Jensen Shannon Divergence (JSD)
def JSD_fromHist(hist1, hist2):
    frq1 = hist1[0]
    frq2 = hist2[0]
    if len(frq1) != len(frq2):
        print('ERROR JSD: Histogram bins not fitting!!')
    return dist.jensenshannon(frq1, frq2)

def JSD_fromHistList_mean(hist_real_list, hist_fake_list):
    JSD_sum = 0.0
    for i in range(len(hist_real_list)):
        frq1 = hist_real_list[i][0]
        frq2 = hist_fake_list[i][0]
        if len(frq1) != len(frq2):
            print('ERROR JSD: Histogram bins not fitting!!')
        JSD_sum += dist.jensenshannon(frq1, frq2)
    return JSD_sum / len(hist_real_list)

# from JSD list
def JSD_weighted_mean(JSD_metrices_list):
    JSD_weighted_list = []
    for i in range(len(JSD_metrices_list)):
        JSD_ary = np.array(JSD_metrices_list[i])
        JSD_weighted = JSD_ary * 1/np.mean(JSD_ary)
        if i == 0:
            print(JSD_weighted)
        JSD_weighted_list.append(JSD_weighted)
    JSD_weighted_mean = np.array(JSD_weighted_list).mean(axis=0)
    return JSD_weighted_mean


## Area Difference a la Anatolii
def AD_fromHist(hist1, hist2):
    frq_1 = hist1[0]
    edges_1 = hist1[1]
    frq_2 = hist2[0]
    edges_2 = hist2[1]
    bin_width = np.diff(edges_1).mean()
    hist_1_norm = frq_1 / sum(frq_1 * bin_width) #normed to integral over area equals 1
    hist_2_norm = frq_2 / sum(frq_2 * bin_width)
    if edges_1.all() == edges_2.all():
        bin_width = np.diff(edges_1).mean()
        diff = abs(hist_1_norm-hist_2_norm)
        return sum(diff * bin_width) * 0.5 # difference area * 1/2 bc integral over 2 hists equals 2
    else:
        return print('ERROR: bin_edges do not align!')
                   
def AD_fromHistList_mean(hist_real_list, hist_fake_list):   # for single_N variables
    AD_sum = 0.0
    for i in range(len(hist_real_list)):
        hist1 = hist_real_list[i]
        hist2 = hist_fake_list[i]
        AD_sum += AD_fromHist(hist1, hist2)
    return AD_sum / len(hist_real_list)

                   
# calculates sigma in a generic CALICE style (performance of physics prototype in 2008 with electrons)
def calcECALsigma(energy_in_GeV):
    a = 0.166
    b = 0.011
    return a**2 + b**2 * energy_in_GeV  

## RMS90 as RMS of smallest window where still 90 percent of events are contained
## in pratice: calculate RMS of all 90% windows, then take window with lowest RMS
def calc90(x):
    x = np.sort(x)
    n10percent = int(round(len(x)*0.1))
    n90percent = len(x) - n10percent
    for i in range(n10percent):
        rms90_i = np.std(x[i:i+n90percent])
        if i == 0:
            rms90 = rms90_i
            mean90 = np.mean(x[i:i+n90percent])
            mean90_err = rms90/np.sqrt(n90percent)
            rms90_err = rms90/np.sqrt(2*n90percent)   # estimator in root
        elif i > 0 and rms90_i < rms90:
            rms90 = rms90_i
            mean90 = np.mean(x[i:i+n90percent])
            mean90_err = rms90/np.sqrt(n90percent)
            rms90_err = rms90/np.sqrt(2*n90percent)   # estimator in root
    return mean90, rms90, mean90_err, rms90_err


# radial hit distribution over all data points
def getEventRadius(data, xbins=30, ybins=30):
    current = np.reshape(data,[-1, xbins,ybins])
    #current_sum = np.sum(current, axis=0)
 
    r_list=[]
    phi_list=[]
    e_list=[]
    n_cent_x = (xbins-1)/2.0
    n_cent_y = (ybins-1)/2.0

    for n_data in range(data.shape[0]):
        r_hit_list = []
        for n_x in np.arange(0, xbins):
            for n_y in np.arange(0, ybins):
                if current[n_data, n_x, n_y] != 0.0:
                    r = np.sqrt((n_x - n_cent_x)**2 + (n_y - n_cent_y)**2)
                    r_hit_list.append(r)
        r_list.append(np.asarray(r_hit_list).mean())
                    #phi = np.arctan((n_x - n_cent_x)/(n_y - n_cent_y))
                    #phi_list.append(phi)
                    #e_list.append(current_sum[n_x,n_y])
                
    r_arr = np.asarray(r_list)
    #phi_arr = np.asarray(phi_list)
    #e_arr = np.asarray(e_list)

    return r_arr#, phi_arr, e_arr

# radial hit distribution over all data points for 3D
def getEventRadius_3d(data, xbins=30, ybins=30, zbins=30):
    current = np.reshape(data,[-1, xbins,ybins,zbins])
    #current_sum = np.sum(current, axis=0)
 
    r_list=[]
    n_cent_x = (xbins-1)/2.0
    n_cent_y = (ybins-1)/2.0
    n_cent_z = (zbins-1)/2.0

    for n_data in range(data.shape[0]):
        r_hit_list = []
        for n_x in np.arange(0, xbins):
            for n_y in np.arange(0, ybins):
                for n_z in np.arange(0, zbins):
                    if current[n_data, n_x, n_y, n_z] != 0.0:
                        r = np.sqrt((n_x - n_cent_x)**2 + (n_y - n_cent_y)**2 + (n_z - n_cent_z)**2)
                        r_hit_list.append(r)

        r_list.append(np.asarray(r_hit_list).mean())
    r_arr = np.asarray(r_list)
    return r_arr#, phi_arr, e_arr




### DATA HANDELING ###

def get_data_loader(input_path, numEvents, batch_size=128, shuffle=True):
    loader_params = {'shuffle': shuffle, 'num_workers': 1, 'pin_memory': True}
    dataset = loader.HDF5Dataset(input_path, transform=tf_logscale, train_size=numEvents)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, **loader_params)
    return data_loader

# returns real images
def getRealImages(filepath, number):
    tf = lambda x:np.sum(x, axis=0)
    dataset_physeval = HDF5Dataset(filepath, transform=tf, train_size=number)
    data = dataset_physeval.get_data_range_tf(0, number)
    energy = dataset_physeval.get_energy_range(0, number)
    return data, energy

# returns 3D real images
def getRealImages_3d(filepath, number):
    dataset_physeval = HDF5Dataset(filepath, train_size=number)
    data = dataset_physeval.get_data_range(0, number)
    energy = dataset_physeval.get_energy_range(0, number)
    return data, energy


# get lists of discrete real data 
def getRealImages_SE_list(save_location, singleE_list, numberSE):
    globalN = int(9 * numberSE * 1.5)
    [glob_real_data_ME, glob_real_Energy_ME, glob_real_data_ME_uncut] = Generate_Plots.getRealImages(save_location, globalN)
    realD_SE_list, realE_SE_list, realD_SE_uncut_list = [], [], []
    for e in singleE_list:
        mask = glob_real_Energy_ME == e 
        realD = glob_real_data_ME[mask[:,0]][0:numberSE]
        realE = glob_real_Energy_ME[mask[:,0]][0:numberSE]
        realD_uncut = glob_real_data_ME_uncut[mask[:,0]][0:numberSE]
        
        realD_SE_list.append(realD)
        realE_SE_list.append(realE)
        realD_SE_uncut_list.append(realD_uncut)
    return realD_SE_list, realE_SE_list, realD_SE_uncut_list



### MODEL FUNCTIONS ###

# give model, discriminator model, and data loader to get mean of discriminator loss
def getLossD_mean(model, netD, data_loader, batch_size, latent_dim, device='cpu'):
    lossD_ary = np.array([])
    with torch.no_grad():
        for i, (data, energy) in enumerate(data_loader): # loading data (1,30,30) and energy in GeV
            data = data.to(device)
            energy = E_true_trans(energy)  # to make energy to 1 GeV = 1e-2
    
            latent = torch.cat((torch.randn(batch_size, latent_dim), energy), dim=1)
            genData = model.decode(latent.to(device))
    
            realFakeData = torch.cat((data.view(-1,1,30,30), genData.view(-1,1,30,30)), dim=1) # concat real data and generated images
            lossD_batch = netD(realFakeData.to(device), energy.to(device)).cpu().numpy()
            #print(lossD_batch.shape)
            lossD_ary = np.append(lossD_ary, lossD_batch)

    return lossD_ary.mean()




### GENERATE IMAGES ###

# generates fake images
def getFakeImagesVAE_ENR(model, number, E_min, E_max, latent_dim, device='cpu', batchsize = 1000):
    fake_list = []
    latent_list=[]
   
    model.eval()   # evaluation mode (no dropout = 0 etc)

    for i in np.arange(0, number, batchsize):
        with torch.no_grad():   # no autograd differentiation done, speeds things up
            latent = torch.cat((torch.randn(batchsize, latent_dim), (torch.rand(batchsize, 1)*(E_max-E_min)+E_min)/1.0), dim=1).to(device)
            data = model.decode(latent).cpu().numpy()     # deconding on device, then sending data to cpu, then converting to numpy

        data = tf_logscale_rev_np(data)     # network output is in logscale, need to reverse this by exp
        fake_list.append(data)
        latent_list.append(latent.cpu())
        print(i)

    # conversion of list to numpy
    data_full = np.vstack(fake_list)
    latent_full = np.vstack(latent_list)
    print(data_full.shape)

    return data_full, latent_full


# generate images from list of energies (in transformed energy range)
def getFakeImagesVAE_ENR_fromEnergyList(model, number, batchsize, E_true_list, latent_dim, device='cpu'):
    fake_list = []
    latent_list=[]
   
    model.eval()   # evaluation mode (no dropout = 0 etc)

    for i in np.arange(0, number, batchsize):
        with torch.no_grad():   # no autograd differentiation done, speeds things up
            latent = torch.cat((torch.randn(batchsize, latent_dim), E_true_list), dim=1).to(device)
            data = model.decode(latent).cpu().numpy()     # deconding on device, then sending data to cpu, then converting to numpy

        data = tf_logscale_rev_np(data)     # network output is in logscale, need to reverse this by exp
        fake_list.append(data)
        latent_list.append(latent.cpu())
        print(i)

    # conversion of list to numpy
    data_full = np.vstack(fake_list)
    latent_full = np.vstack(latent_list)
    print(data_full.shape)

    return data_full, latent_full


# returns exatly one image based on a set of already sampled latence space variables - for 2D
def getOneFakeImageVAE_ENR(model, latent_space, energy, device):
    model.eval()
    with torch.no_grad():
        latent = torch.cat((latent_space, energy), dim=1).to(device)
        image = model.decode(latent).cpu().numpy()
    image = tf_logscale_rev_np(image) # reverse the log scaling
    return image 


### 3D 

## reconstruct 3d images based on true energy list and a list of already sampled or determined latent variables z
def reconstructImages_3d_fromZ(model, E_true_list, z_list, batchsize = 200, device='cpu'):
    model.eval()   # evaluation mode (no dropout = 0 etc)

    energy = torch.tensor(E_true_list).to(device)#.type(torch.cuda.DoubleTensor)
    z = torch.tensor(z_list).to(device)  # needs to be dtype ('<f4')

    with torch.no_grad():   # no autograd differentiation done, speeds things up
        reco = model.module.decode(torch.cat((z, energy), 1)).cpu().numpy()     # deconding on device, then sending data to cpu, then converting to numpy

    reco = reco.reshape(-1,30,30,30)

    return reco


def reconstructImages_3d_fromZ_PP(model, model_P, E_true_list, z_list, batchsize = 200, device='cpu', thresh=0.1):
    model.eval()   # evaluation mode (no dropout = 0 etc)

    energy = torch.tensor(E_true_list).to(device)#.type(torch.cuda.DoubleTensor)
    z = torch.tensor(z_list).to(device)  # needs to be dtype ('<f4')

    with torch.no_grad():   # no autograd differentiation done, speeds things up
        reco = model.module.decode(torch.cat((z, energy), 1))  
        reco = model_P.forward(reco, energy).cpu().numpy()    # POST PROCESSING

    # apply MIP cut
    reco[reco < thresh] = 0.0
        
    reco = reco.reshape(-1,30,30,30)
    return reco



## reconstruct 3d images based on true energy list and mu & sigma list 
def reconstructImages_3d_fromMuStd(model, E_true_list, mu_list, std_list, batchsize = 200, device='cpu'):
    model.eval()   # evaluation mode (no dropout = 0 etc)

    energy = torch.tensor(E_true_list).to(device)#.type(torch.cuda.DoubleTensor)
    mu = torch.tensor(mu_list).to(device)    # needs to be dtype ('<f4')
    std = torch.tensor(std_list).to(device)  # needs to be dtype ('<f4')
    logvar = torch.log(2.*std)

    with torch.no_grad():   # no autograd differentiation done, speeds things up
        z = model.module.reparameterize(mu, logvar)
        reco = model.module.decode(torch.cat((z, energy), 1)).cpu().numpy()     # deconding on device, then sending data to cpu, then converting to numpy

    reco = reco.reshape(-1,30,30,30)
    z = z.cpu().numpy()

    return reco, z




# reconstruct (!) one batch of 3d images with E_true list in GeV - either with mean or reparametrized
def reconstructImages_3d_batch(model, real_images, E_true_list, batchsize = 200, latent_dim=512, device='cpu', thresh=0.1):
    model.eval()   # evaluation mode (no dropout = 0 etc)
    energy_list = torch.tensor(E_true_list).to(device)
    real_images = torch.tensor(real_images).to(device)
    zeros = torch.zeros(batchsize,latent_dim).to(device)
    with torch.no_grad():   # no autograd differentiation done, speeds things up
        mu, logvar = model.module.encode(real_images.float(), energy_list.float())
        std = torch.exp(0.5*logvar)
        z = model.module.reparameterize(mu, logvar)
        z_mu0 = model.module.reparameterize(zeros, logvar)
        z_std1 = model.module.reparameterize(mu, zeros)
        reco_rand = model.module.decode(torch.cat((z, energy_list), 1)).cpu().numpy()     # deconding on device, then sending data to cpu, then converting to numpy
        reco_mu0 = model.module.decode(torch.cat((z_mu0, energy_list), 1)).cpu().numpy()     # deconding on device, then sending data to cpu, then converting to numpy
        reco_std1 = model.module.decode(torch.cat((z_std1, energy_list), 1)).cpu().numpy()     # deconding on device, then sending data to cpu, then converting to numpy
        reco_wMean = model.module.decode(torch.cat((mu, energy_list), 1)).cpu().numpy()     # deconding on device, then sending data to cpu, then converting to numpy
        
    # apply MIP cut
    reco_rand[reco_rand < thresh] = 0.0
    reco_mu0[reco_mu0 < thresh] = 0.0
    reco_std1[reco_std1 < thresh] = 0.0
    reco_wMean[reco_wMean < thresh] = 0.0

    reco_rand = reco_rand.reshape(-1,30,30,30)
    reco_mu0 = reco_mu0.reshape(-1,30,30,30)
    reco_std1 = reco_std1.reshape(-1,30,30,30)
    reco_wMean = reco_wMean.reshape(-1,30,30,30)
    mu = mu.cpu().numpy()
    std = std.cpu().numpy()
    z = z.cpu().numpy()
    z_mu0 = z_mu0.cpu().numpy()
    z_std1 = z_std1.cpu().numpy()

    return reco_rand, reco_mu0, reco_std1, reco_wMean, mu, std, z, z_mu0, z_std1


def reconstructImages_3d_batch_PP(model, model_P, real_images, E_true_list, batchsize = 200, latent_dim=512, device='cpu', thresh=0.1):
    model.eval()   # evaluation mode (no dropout = 0 etc)
    model_P.eval()
    energy_list = torch.tensor(E_true_list).to(device)
    real_images = torch.tensor(real_images).to(device)
    zeros = torch.zeros(batchsize,latent_dim).to(device)
    with torch.no_grad():   # no autograd differentiation done, speeds things up
        mu, logvar = model.module.encode(real_images.float(), energy_list.float())
        std = torch.exp(0.5*logvar)
        z = model.module.reparameterize(mu, logvar)
        z_mu0 = model.module.reparameterize(zeros, logvar)
        z_std1 = model.module.reparameterize(mu, zeros)
        
        reco_rand = model.module.decode(torch.cat((z, energy_list), 1))    # deconding on device, then sending data to cpu, then converting to numpy
        reco_rand = model_P.forward(reco_rand, energy_list).cpu().numpy()    # POST PROCESSING
        
        reco_mu0 = model.module.decode(torch.cat((z_mu0, energy_list), 1))   # deconding on device, then sending data to cpu, then converting to numpy
        reco_mu0 = model_P.forward(reco_mu0, energy_list).cpu().numpy()   # PP
        
        reco_std1 = model.module.decode(torch.cat((z_std1, energy_list), 1))    # deconding on device, then sending data to cpu, then converting to numpy
        reco_std1 = model_P.forward(reco_std1, energy_list).cpu().numpy()   # PP
        
        reco_wMean = model.module.decode(torch.cat((mu, energy_list), 1))     # deconding on device, then sending data to cpu, then converting to numpy
        reco_wMean = model_P.forward(reco_wMean, energy_list).cpu().numpy()   # PP
        
    # apply MIP cut
    reco_rand[reco_rand < thresh] = 0.0
    reco_mu0[reco_mu0 < thresh] = 0.0
    reco_std1[reco_std1 < thresh] = 0.0
    reco_wMean[reco_wMean < thresh] = 0.0

    reco_rand = reco_rand.reshape(-1,30,30,30)
    reco_mu0 = reco_mu0.reshape(-1,30,30,30)
    reco_std1 = reco_std1.reshape(-1,30,30,30)
    reco_wMean = reco_wMean.reshape(-1,30,30,30)
    mu = mu.cpu().numpy()
    std = std.cpu().numpy()
    z = z.cpu().numpy()
    z_mu0 = z_mu0.cpu().numpy()
    z_std1 = z_std1.cpu().numpy()

    return reco_rand, reco_mu0, reco_std1, reco_wMean, mu, std, z, z_mu0, z_std1



# generate one batch of 3d images with E_true list in GeV
def getFakeImages_batch_3d(model, E_true_list, batchsize = 1000, latent_dim=512, device='cpu', thresh=0.1):
    model.eval()   # evaluation mode (no dropout = 0 etc)
    energy_list = torch.tensor(E_true_list).to(device)
    with torch.no_grad():   # no autograd differentiation done, speeds things up
        latent_space = torch.randn(batchsize, latent_dim).to(device)
        latent_space_wEnergy = torch.cat((latent_space, energy_list), dim=1).to(device)
        data = model.module.decode(latent_space_wEnergy).cpu().numpy()     # deconding on device, then sending data to cpu, then converting to numpy

    # apply MIP cut
    data[data < thresh] = 0.0
    data = data.reshape(-1,30,30,30)
    latent_space = latent_space.cpu().numpy()

    return data, latent_space

# generate a full number of many batches of fake images from E_true_list in GeV
def getFakeImages_3d_noPP_all(model, real_energy, numEvents, batchsize=200, latent_dim=512, device='cpu', thresh=0.1):
    fake_list = []
    latent_list = []
    for i in range(0, numEvents, batchsize):
        energylist = real_energy[i:i+batchsize]
        fake_batch, latent_batch = getFakeImages_batch_3d(model, energylist, batchsize, latent_dim, device, thresh)
        fake_list.append(fake_batch)
        latent_list.append(latent_batch)
    fake_images_noPP = np.vstack(fake_list)
    z_fake = np.vstack(latent_list)
    return fake_images_noPP, z_fake

# create a list of fake images from a list of single energies
def getFakeImages_3d_noPP_singleEnergies(model, singleE_list, numEvents, batchsize=200, latent_dim=512, device='cpu', thresh=0.1):
    fake_images_noPP_SE_list, z_fake_SE_list = [], []
    for se in range(len(singleE_list)):
        constE = np.full((numEvents,1), singleE_list[se], '<f4')
        fake_images_noPP_SE, z_fake_SE = getFakeImages_3d_noPP_all(model, constE, numEvents, batchsize, latent_dim, device, thresh)
        fake_images_noPP_SE_list.append(fake_images_noPP_SE)
        z_fake_SE_list.append(z_fake_SE)
    return fake_images_noPP_SE_list, z_fake_SE_list    





# FOR POST PROCESSING
# generate one batch of 3d images with E_true list in GeV
def getFakeImages_batch_3d_PP(model, model_P, E_true_list, batchsize = 1000, latent_dim=512, device='cpu', thresh=0.1):
    model.eval()   # evaluation mode (no dropout = 0 etc)
    model_P.eval()
    energy_list = torch.tensor(E_true_list).to(device)
    with torch.no_grad():   # no autograd differentiation done, speeds things up
        latent_space = torch.randn(batchsize, latent_dim).to(device)
        latent_space_wEnergy = torch.cat((latent_space, energy_list), dim=1).to(device)
        data_noPP = model.module.decode(latent_space_wEnergy)     # deconding on device, then sending data to cpu, then converting to numpy
        data = model_P.forward(data_noPP, energy_list).cpu().numpy()    # POST PROCESSING
        
    # apply MIP cut
    data[data < thresh] = 0.0
    data = data.reshape(-1,30,30,30)
    latent_space = latent_space.cpu().numpy()

    return data, latent_space



# FOR POST PROCESSING: generate a full number of many batches of fake images from E_true_list in GeV
def getFakeImages_3d_PP_all(model, model_P, real_energy, numEvents, batchsize=200, latent_dim=512, device='cpu', thresh=0.1):
    fake_list = []
    latent_list = []
    for i in range(0, numEvents, batchsize):
        energylist = real_energy[i:i+batchsize]
        fake_batch, latent_batch = getFakeImages_batch_3d_PP(model, model_P, energylist, batchsize, latent_dim, device, thresh)
        fake_list.append(fake_batch)
        latent_list.append(latent_batch)
    fake_images_noPP = np.vstack(fake_list)
    z_fake = np.vstack(latent_list)
    return fake_images_noPP, z_fake

# create a list of fake images from a list of single energies
def getFakeImages_3d_PP_singleEnergies(model, model_P, singleE_list, numEvents, batchsize=200, latent_dim=512, device='cpu', thresh=0.1):
    fake_images_noPP_SE_list, z_fake_SE_list = [], []
    for se in range(len(singleE_list)):
        constE = np.full((numEvents,1), singleE_list[se], '<f4')
        fake_images_noPP_SE, z_fake_SE = getFakeImages_3d_PP_all(model, model_P, constE, numEvents, batchsize, latent_dim, device, thresh)
        fake_images_noPP_SE_list.append(fake_images_noPP_SE)
        z_fake_SE_list.append(z_fake_SE)
    return fake_images_noPP_SE_list, z_fake_SE_list   





### WITH KDE SAMPLING FOR POST PROCESSING: generate a full number of many batches of fake images from E_true_list in GeV

# generate one batch of 3d images with E_true list in GeV
def getFakeImages_batch_3d_PP_KDE_sampling(kde_kernel_file, model, model_P, E_true_list, batchsize = 1000, latent_dim=512, device='cpu', thresh=0.1, latent_sml=24):
    model.eval()   # evaluation mode (no dropout = 0 etc)
    model_P.eval()
    energy_list = torch.tensor(E_true_list)#.to(device)
    
    # import kde kernel via pickle = dill
    with open(kde_kernel_file, 'rb') as f:
        kde = pickle.load(f)
        
    with torch.no_grad():   # no autograd differentiation done, speeds things up        
        kde_sample = kde.resample(batchsize).T
        normal_sample = torch.randn(batchsize, latent_dim - latent_sml)
        z = torch.cat((torch.from_numpy(np.float32(kde_sample)), normal_sample), dim=1)
        
        latent_space_wEnergy = torch.cat((z, energy_list), dim=1).to(device)
        data_noPP = model.module.decode(latent_space_wEnergy)     # deconding on device, then sending data to cpu, then converting to numpy
        data = model_P.forward(data_noPP, energy_list).cpu().numpy()    # POST PROCESSING
        
    # apply MIP cut
    data[data < thresh] = 0.0
    data = data.reshape(-1,30,30,30)
    latent_space = z.cpu().numpy()

    return data, latent_space


def getFakeImages_3d_PP_all_KDE_sampling(kde_kernel_file, model, model_P, real_energy, numEvents, batchsize=200, latent_dim=512, device='cpu', thresh=0.1, latent_sml=24):
    fake_list = []
    latent_list = []
    for i in range(0, numEvents, batchsize):
        energylist = real_energy[i:i+batchsize]
        fake_batch, latent_batch = getFakeImages_batch_3d_PP_KDE_sampling(kde_kernel_file, model, model_P, energylist, batchsize, latent_dim, device, thresh, latent_sml)
        fake_list.append(fake_batch)
        latent_list.append(latent_batch)
    fake_images_noPP = np.vstack(fake_list)
    z_fake = np.vstack(latent_list)
    return fake_images_noPP, z_fake

# create a list of fake images from a list of single energies
def getFakeImages_3d_PP_singleEnergies_KDE_sampling(kde_kernel_file, model, model_P, singleE_list, numEvents, batchsize=200, latent_dim=512, device='cpu', thresh=0.1, latent_sml=24):
    fake_images_noPP_SE_list, z_fake_SE_list = [], []
    for se in range(len(singleE_list)):
        constE = np.full((numEvents,1), singleE_list[se], '<f4')
        fake_images_noPP_SE, z_fake_SE = getFakeImages_3d_PP_all_KDE_sampling(kde_kernel_file, model, model_P, constE, numEvents, batchsize, latent_dim, device, thresh, latent_sml)
        fake_images_noPP_SE_list.append(fake_images_noPP_SE)
        z_fake_SE_list.append(z_fake_SE)
    return fake_images_noPP_SE_list, z_fake_SE_list   



#### KDE E --> KDE sampling with energy
### WITH KDE SAMPLING FOR POST PROCESSING: generate a full number of many batches of fake images from E_true_list in GeV

# generate one batch of 3d images with E_true list in GeV
def getFakeImages_batch_3d_PP_KDE_sampling_wEnergy(kde_kernel_file, model, model_P, Emin=10, Emax=100, batchsize = 1000, latent_dim=512, device='cpu', thresh=0.1, latent_sml=24):
    model.eval()   # evaluation mode (no dropout = 0 etc)
    model_P.eval()
    #energy_list = torch.tensor(E_true_list)#.to(device)
    
    # import kde kernel via pickle = dill
    with open(kde_kernel_file, 'rb') as f:
        kde = pickle.load(f)
        
    with torch.no_grad():   # no autograd differentiation done, speeds things up     
        latent = []
        E = []
        numberAccepted = 0
        while numberAccepted < batchsize:
            kde_sample = kde.resample(20000).T
            kde_latent = kde_sample[:, :latent_sml]
            kde_energy = kde_sample[:, latent_sml:]

            energy_mask = (kde_energy[:,0] < Emax+0.01) & (kde_energy[:,0] > Emin-0.01)  

            kde_latent_masked = kde_latent[energy_mask]
            kde_energy_masked = kde_energy[energy_mask]


            numberAccepted += kde_latent_masked.shape[0]

            normal_sample = torch.randn(kde_latent_masked.shape[0], latent_dim - latent_sml)
            latent.append(torch.cat((torch.from_numpy(np.float32(kde_latent_masked)), normal_sample), dim=1))
            E.append(torch.from_numpy(np.float32(kde_energy_masked)))


        latent = torch.cat((latent)).to(device)[:batchsize]
        E = torch.cat((E), 0).to(device)[:batchsize]
        x = torch.zeros(batchsize, latent_dim, device=device)
        z = latent


        data_noPP = model(x=x, E_true=E, 
                        z = z,  mode='decode')  
        
        latent = torch.cat((z, E), dim=1)

        latent = latent.cpu().numpy()

        data = model_P.forward(data_noPP, E).cpu().numpy()
        
        #kde_sample = kde.resample(batchsize).T
        #normal_sample = torch.randn(batchsize, latent_dim - latent_sml)
        #z = torch.cat((torch.from_numpy(np.float32(kde_sample)), normal_sample), dim=1)
        
        #latent_space_wEnergy = torch.cat((z, energy_list), dim=1).to(device)
        #data_noPP = model.module.decode(latent_space_wEnergy)     # deconding on device, then sending data to cpu, then converting to numpy
        #data = model_P.forward(data_noPP, energy_list).cpu().numpy()    # POST PROCESSING
        
    # apply MIP cut
    data[data < thresh] = 0.0
    data = data.reshape(-1,30,30,30)
    #latent_space = z.cpu().numpy()

    return data, latent


def getFakeImages_3d_PP_all_KDE_sampling_wEnergy(kde_kernel_file, model, model_P, real_energy, numEvents, batchsize=200, latent_dim=512, device='cpu', thresh=0.1, latent_sml=24):
    fake_list = []
    latent_list = []
    Emin, Emax = real_energy.min(), real_energy.max()
    for i in range(0, numEvents, batchsize):
        #energylist = real_energy[i:i+batchsize]
        fake_batch, latent_batch = getFakeImages_batch_3d_PP_KDE_sampling_wEnergy(kde_kernel_file, model, model_P, Emin, Emax, batchsize, latent_dim, device, thresh, latent_sml)
        fake_list.append(fake_batch)
        latent_list.append(latent_batch)
    fake_images_noPP = np.vstack(fake_list)
    z_fake = np.vstack(latent_list)
    return fake_images_noPP, z_fake


# create a list of fake images from a list of single energies
def getFakeImages_3d_PP_singleEnergies_KDE_sampling_wEnergy(kde_kernel_file, model, model_P, singleE_list, numEvents, batchsize=200, latent_dim=512, device='cpu', thresh=0.1, latent_sml=24):
    fake_images_noPP_SE_list, z_fake_SE_list = [], []
    for se in range(len(singleE_list)):
        constE = np.full((numEvents,1), singleE_list[se], '<f4')
        fake_images_noPP_SE, z_fake_SE = getFakeImages_3d_PP_all_KDE_sampling_wEnergy(kde_kernel_file, model, model_P, constE, numEvents, batchsize, latent_dim, device, thresh, latent_sml)
        fake_images_noPP_SE_list.append(fake_images_noPP_SE)
        z_fake_SE_list.append(z_fake_SE)
    return fake_images_noPP_SE_list, z_fake_SE_list   







# encode (!) one batch of 3d images with E_true list in GeV
def encodeImages_3d_batch(model, real_images, real_energy, batchsize = 200, device='cpu'):
    model.eval()   # evaluation mode (no dropout = 0 etc)
    energy_list = torch.tensor(real_energy).to(device)
    real_images = torch.tensor(real_images).to(device)
    with torch.no_grad():   # no autograd differentiation done, speeds things up
        mu, logvar = model.module.encode(real_images.float(), energy_list.float())
        std = torch.exp(0.5*logvar)
    mu = mu.cpu().numpy()
    std = std.cpu().numpy()
    return mu, std

# same thing, but for multiple batches
def encodeImages_3d_all(model, real_images, real_energy, nEvents, batchsize=200, device='cpu'):
    mu_list, std_list = [], []
    for i in range(0, nEvents, batchsize):
        energy_batch = real_energy[i:i+batchsize]
        image_batch = real_images[i:i+batchsize]
        mu, std = encodeImages_3d_batch(model, image_batch, energy_batch, batchsize, device)
        mu_list.append(mu)
        std_list.append(std)
    encoder_mu = np.vstack(mu_list)
    encoder_std = np.vstack(std_list)
    return encoder_mu, encoder_std






### PLOTS ###

def plt_ExampleImage(image, model_title='ML Model', save_title='ML_model', mode='comp'):
    figExIm = plt.figure(figsize=(6,6))
    axExIm = figExIm.add_subplot(1,1,1)

    #image = tf_logscale_rev_np(np.reshape(image,(1,30,30)))+1.0
    image = (np.reshape(image,(1,30,30)))+1.0
    #print(np.min(image))
    
    masked_array = np.ma.array(image, mask=(image==1.0))
    cmap = mpl.cm.viridis
    cmap.set_bad('white',1.)

    im = axExIm.imshow(masked_array[0], filternorm=False, interpolation='none', cmap = cmap, vmin=1, vmax=100,
                       norm=mpl.colors.LogNorm())
    figExIm.patch.set_facecolor('white')
    axExIm.title.set_text(model_title)
    figExIm.colorbar(im)
    plt.show()
    #plt.savefig('./plots/' + save_title+".png")
 


# hit energy histogram (only real hits, no zero values)
def plt_hitEnergyDist_modelList(i, model_list, real_data_list, fake_data_list, latent_list, energy_list, hl_latent_list):

    #### plot needs: real_data_list, fake_data_list, model_list, latent_list, energy_list, hl_latent_list
    plt.clf()
    plt.figure(figsize=(12,6))
    
    # real data
    hist_data = real_data_list[i].flatten()
    hist_data = hist_data[hist_data>0.]       # ignore all zero entries
    x_max = np.max(hist_data)*1.1
    hist_real = plt.hist(hist_data, bins=100, histtype='step', range=[0,x_max], label='Geant4', linewidth=3)
    
    # all the generated data
    for j in range(len(model_list)):
        if latent_list[j] in hl_latent_list:
            linewidth = 3
        else:
            linewidth = 0.5
        hist_data = fake_data_list[i][j].flatten()
        hist_data = hist_data[hist_data>0.]       # ignore all zero entries
        label = 'latent: {0}'.format(latent_list[j])
        hist = plt.hist(hist_data, bins=hist_real[1], histtype='step', range=[0,x_max], label=label, linewidth=linewidth)
    
    plt.yscale('log')
    plt.title('{0}GeV: hit energy distribution'.format(energy_list[i]), fontsize=18)
    plt.legend(loc='upper right', fontsize=15)
    plt.xlabel('hit energy [MeV]', fontsize=14)
    plt.ylabel('hits', fontsize=14)
    plt.show()  


### RADIAL DISTRIBUTION PLOT: EVENT RADIUS
def plt_eventRadiusDist(i, model_list, real_data_list, fake_data_list, latent_list, energy_list, hl_latent_list):
    plt.clf()
    plt.figure(figsize=(12,6))
    
    # real
    r_arr = getEventRadius(real_data_list[i])
    hist_real = plt.hist(r_arr, bins=100, range=[0,15], label='Geant4', linewidth=3, histtype='step')
    
    # fake
    for j in range(len(model_list)):
        r_arr = getEventRadius(fake_data_list[i][j])
        label = 'latent: {0}'.format(latent_list[j])
        if latent_list[j] in hl_latent_list:
            linewidth = 3
        else: 
            linewidth = 0.5
        hist = plt.hist(r_arr, bins=hist_real[1], range=[0,15], label=label, linewidth=linewidth, histtype='step')
    
    plt.xlabel("event radius [pxl]", fontsize = 14)
    plt.ylabel("events", fontsize = 14)
    plt.legend(loc='upper left', fontsize = 15)
    plt.title('{0} GeV: event radius distribution'.format(energy_list[i]), fontsize = 18)
    plt.show()


# energy sum plot
def plt_energySum_modelList(i, model_list, real_data_list, fake_data_list, latent_list, energy_list, hl_latent_list):
    plt.clf()
    plt.figure(figsize=(6,6))
    
    data = real_data_list[i].sum(axis=2).sum(axis=1)
    x_min = np.min(data)*0.9  # x min and max plus minus 10%
    x_max = np.max(data)*1.1
    hist_real = plt.hist(data, bins=50, histtype='step', range=[x_min,x_max], label='Geant4', linewidth=3)
    
    for j in range(len(model_list)):
        if latent_list[j] in hl_latent_list:
            linewidth = 3
        else:
            linewidth = 0.5
        data = fake_data_list[i][j].sum(axis=2).sum(axis=1)
        label='latent: {0}'.format(latent_list[j])
        hist = plt.hist(data, bins=hist_real[1], histtype='step' , range=[x_min,x_max], label=label, linewidth=linewidth)
    
    plt.title('{0} GeV: event energy sum'.format(energy_list[i]), fontsize=18)
    plt.xlabel('energy sum [MeV]', fontsize=14)
    plt.ylabel('events', fontsize=14)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()


# mean90, rms90 linearity and resolution plot form modelList
def plt_linearity_resolution_fromModelList(model_list, real_data_list, fake_data_list, latent_list, energy_list, hl_latent_list):
    ###### first calculate mean90, rms90
    # real data
    mean90_real_list = []
    rms90_real_list = []
    for i in range(len(energy_list)):
        data = real_data_list[i].sum(axis=2).sum(axis=1)
        mean90, rms90, mean90_err, rms90_err = calc90(data)
        mean90_real_list.append(mean90)
        rms90_real_list.append(rms90)
    
    # fake data
    mean90_fake_list = []
    rms90_fake_list = []
    for j in range(len(model_list)):
        mean90_oneModel_list = []
        rms90_oneModel_list = []
        for i in range(len(energy_list)):
            data = fake_data_list[i][j].sum(axis=2).sum(axis=1)
            mean90, rms90, mean90_err, rms90_err = calc90(data)
            mean90_oneModel_list.append(mean90)
            rms90_oneModel_list.append(rms90)
        
        mean90_fake_list.append(mean90_oneModel_list)
        rms90_fake_list.append(rms90_oneModel_list)
    
    ##### LINEARITY PLOT 
    plt.clf()
    plt.figure(figsize=(6,6))
    
    # real
    if latent_list[j] in hl_latent_list:
        linewidth = 3 
    else:
        linewidth = 0.5 
    plt.plot(energy_list, mean90_real_list, label='Geant4', marker='.', linewidth=linewidth )
    
    # fake
    for j in range(len(model_list)):
        if latent_list[j] in hl_latent_list:
            linewidth = 3 
        else:
            linewidth = 0.5 
        label='latent: {0}'.format(latent_list[j])
        plt.plot(energy_list, mean90_fake_list[j], marker='.', label=label, linewidth=linewidth)
    
    plt.title('linearity', fontsize = 18)
    plt.xlabel('beam energy [GeV]', fontsize = 14)
    plt.ylabel('mean90 [MeV]', fontsize = 14)
    plt.legend(loc='upper left', fontsize = 14)
    plt.show()
    
    ### RESOLUTION PLOT
    plt.clf()
    plt.figure(figsize=(6,6))
    
    # real
    if latent_list[j] in hl_latent_list:
        linewidth = 3 
    else:
        linewidth = 0.5 
    plt.plot(energy_list, np.array(rms90_real_list) / np.array(mean90_real_list), label='Geant4', marker='.', linewidth=linewidth )
    
    # fake
    for j in range(len(model_list)):
        if latent_list[j] in hl_latent_list:
            linewidth = 3 
        else:
            linewidth = 0.5 
        label='latent: {0}'.format(latent_list[j])
        plt.plot(energy_list, np.array(rms90_fake_list[j]) / np.array(mean90_fake_list[j]), marker='.', label=label, linewidth=linewidth)
    
    plt.title('resolution', fontsize = 18)
    plt.xlabel('beam energy [GeV]', fontsize = 14)
    plt.ylabel('rms90 / mean90 [MeV]', fontsize = 14)
    plt.legend(loc='upper right', fontsize = 13)
    plt.show()


    
    
    
##### 3d plotting #####
## 3d shower plot based on Saschas 3d plot
def plotMatrix(ax, x, y, z, data, cmap="jet", cax=None, alpha=0.1, edgecolors=None):
    # plot a Matrix 
    norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())
    colors = lambda i : mpl.cm.ScalarMappable(norm=norm,cmap = cmap).to_rgba(data[i]) 
    norm_max = np.max(data)
    for i, xi in enumerate(x):
        l2 = 0.3+0.7*data[i]/norm_max
        plotCubeAtA(pos=(x[i], y[i], z[i]), l=l2, c=colors(i), alpha=alpha,  ax=ax, edgecolors=edgecolors)
      

    
def plotCubeAtA(pos=(0,0,0), l=1.0, c="b", alpha=0.1, ax=None, edgecolors=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        x_range = np.array([[pos[0]-l/2, pos[0]+l/2]])
        y_range = np.array([[pos[1]-l/2, pos[1]+l/2]])
        z_range = np.array([[pos[2]-l/2, pos[2]+l/2]])
        
        z_range
        xx, yy = np.meshgrid(x_range, y_range)
        
        ax.plot_surface(xx, yy, (np.tile(z_range[:,0:1], (2, 2))), color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface(xx, yy, (np.tile(z_range[:,1:2], (2, 2))), color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)

        yy, zz = np.meshgrid(y_range, z_range)
        ax.plot_surface((np.tile(x_range[:,0:1], (2, 2))), yy, zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface((np.tile(x_range[:,1:2], (2, 2))), yy, zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)

        xx, zz = np.meshgrid(x_range, z_range)
        ax.plot_surface(xx, (np.tile(y_range[:,0:1], (2, 2))), zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface(xx, (np.tile(y_range[:,1:2], (2, 2))), zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)

        
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = clr.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plt_3dShower(image, model_title='ML Model', save_title='ML_model', mode='comp'):
    figExIm = plt.figure(figsize=(16,16))
    axExIm1 = figExIm.gca(projection='3d')

    axExIm1.set_xticklabels([])
    axExIm1.set_yticklabels([])
    axExIm1.set_zticklabels([])
    axExIm1.set_xticks(np.arange(0,30,1))
    axExIm1.set_yticks(np.arange(0,30,1))
    axExIm1.set_zticks(np.arange(0,30,1))   
   
    axExIm1.set_xlim([0, 30])
    axExIm1.set_ylim([0, 30])
    axExIm1.set_zlim([0, 30])
    #print(image.shape)
    image = image+1.0
    
    masked_array = np.ma.array(image, mask=(image==1.0))
    cmap = mpl.cm.viridis

    xL,yL,zL,cL = [],[],[],[]
    for index, c in np.ndenumerate(masked_array):
        (x,y,z) = index
        if c != 1:
            xL.append(x)
            yL.append(y)
            zL.append(z)
            cL.append(np.log(c))
            #break

    cmap = mpl.cm.viridis
    cmap.set_bad('white',1.)

    xL = np.array(xL)
    yL = np.array(yL)
    zL = np.array(zL)
    cL = np.array(cL)
    print(xL.shape)
    
    figExIm.patch.set_facecolor('white')

    cmap = mpl.cm.jet
    norm= mpl.colors.LogNorm()
    
    my_cmap = truncate_colormap(cmap, 0.0, 0.7)
    transparent = (0.1, 0.1, 0.9, 0.0)

    axExIm1.set_xticklabels([])
    axExIm1.set_yticklabels([])
    axExIm1.set_zticklabels([])
    axExIm1.set_xticks(np.arange(0,30,1))
    axExIm1.set_yticks(np.arange(0,30,1))
    axExIm1.set_zticks(np.arange(0,30,1))    

    axExIm1.set_title(model_title, y=1.04, fontsize=70)
    
    plotMatrix(ax=axExIm1, x=xL, y=yL, z=zL, data=cL, cmap=my_cmap, alpha=0.7, edgecolors=transparent)
    
    plt.savefig('./plots/' + save_title+".png", bbox_inches='tight', transparent=True)
    return axExIm1



### CORRELATION PLOTS

def runCorrelations_data_latent(data, energy, latent_space, latent_names, Model_title='ML Model', save_title='ML_model', xbins=30, ybins=30, layers=30, enable_name_list=True):
    #First calculate relevant parameters from shower
    import seaborn as sns
    import pandas
    #Moments data[B,Z,X,Y]
    nMax = 20000
    data = data[0:nMax]
    latent_space = latent_space[0:nMax]
    energy = energy[0:nMax]
    
    name_list = []
    data_list = []
    
    ### CALCULATE METRICES FROM DATA

    Moment_0_x = Generate_Plots.get0Moment(np.sum(data, axis=(1,3)))
    Moment_0_y = Generate_Plots.get0Moment(np.sum(data, axis=(1,2)))
    Moment_0_z = Generate_Plots.get0Moment(np.sum(data, axis=(2,3)))
    data_list.append(Moment_0_x)
    data_list.append(Moment_0_y)
    data_list.append(Moment_0_z)    
    name_list.append('$\mathrm{m}_{1, \mathrm{x}}$')
    name_list.append('$\mathrm{m}_{1, \mathrm{y}}$')
    name_list.append('$\mathrm{m}_{1, \mathrm{z}}$')

    Moment_1_x = Generate_Plots.get1Moment(np.sum(data, axis=(1,3)))
    Moment_1_y = Generate_Plots.get1Moment(np.sum(data, axis=(1,2)))
    Moment_1_z = Generate_Plots.get1Moment(np.sum(data, axis=(2,3)))
    data_list.append(Moment_1_x)
    data_list.append(Moment_1_y)
    data_list.append(Moment_1_z)    
    #print(Moment_1_z.shape)
    name_list.append('$\mathrm{m}_{2, \mathrm{x}}$')
    name_list.append('$\mathrm{m}_{2, \mathrm{y}}$')
    name_list.append('$\mathrm{m}_{2, \mathrm{z}}$')

    ecal_sum = Generate_Plots.getTotE(data)
    data_list.append(ecal_sum)    
    name_list.append('$\mathrm{E}_{\\mathrm{vis}}$')
    #print(ecal_sum.shape)
    
    p_energy = energy[0:nMax, 0]
    data_list.append(p_energy)    
    name_list.append('$\mathrm{E}_{\\mathrm{inc}}$')
    #print(p_energy)

    hits = Generate_Plots.getOcc(data)
    data_list.append(hits)    
    name_list.append('$\mathrm{n}_{\\mathrm{hit}}$')
    #print(hits.shape)

    ratio1_total = np.sum(data[:,0:10], axis=(1,2,3))/ecal_sum
    ratio2_total = np.sum(data[:,10:20], axis=(1,2,3))/ecal_sum
    ratio3_total = np.sum(data[:,20:30], axis=(1,2,3))/ecal_sum
    data_list.append(ratio1_total)    
    data_list.append(ratio2_total)    
    data_list.append(ratio3_total)    
    #print(ratio1_total.shape)

    name_list.append('$\mathrm{E}_{1}/\mathrm{E}_{\\mathrm{vis}}$')
    name_list.append('$\mathrm{E}_{2}/\mathrm{E}_{\\mathrm{vis}}$')
    name_list.append('$\mathrm{E}_{3}/\mathrm{E}_{\\mathrm{vis}}$')
    
    ### GET DATAFRAMES OF LATENT SPACE and DATA
    df_data = pandas.DataFrame(data=np.vstack(data_list).transpose(), columns = name_list)
    df_latent = pandas.DataFrame(latent_space, columns=latent_names)
    df_energy = pandas.DataFrame(energy, columns=['E'])
    
    ### MAKE CORRELATION PLOT
    df = pandas.concat([df_data,df_latent,df_energy], axis=1) 
    correlations_full = df.corr().iloc[0 : df_data.shape[1], df_data.shape[1] : df.shape[1]]
    
    #upper_triangle = np.ones((len(name_list),len(name_list)))
    #upper_triangle[np.triu_indices(len(name_list),1)] = 0.0
    #print(upper_triangle)
    
    correlations = correlations_full.mask((correlations_full < 0.05) & (correlations_full > -0.05))#.mask(upper_triangle == 0)
    correlations = np.abs(correlations)
    #correlations.mask(correlations < 0.5) == 0
    
    fig_cor = plt.figure(figsize=(10,10), dpi=200)
    ax_cor = fig_cor.add_subplot(1,1,1)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)

    temp1 = plt.rcParams['font.size']
    temp2 = plt.rcParams['font.family']
    
    plt.rcParams['font.size'] = 50
    plt.rcParams['font.family'] = "serif"
    
    if enable_name_list:
        ticks_y = correlations.index
    else:
        ticks_y = False
        
    g = sns_plot = sns.heatmap((correlations),
        xticklabels=correlations.columns,
        yticklabels=ticks_y,
        cmap=sns.diverging_palette(250, 250, as_cmap=True),
        annot=True, ax=ax_cor, vmin=-1, vmax=1,
        annot_kws={"size": 17}, fmt=".1f", square=True, cbar=False, linewidths=0, linecolor='black'
        )

    g.tick_params(axis='both', which='both', length=0)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 20, rotation='horizontal')
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 20, rotation='horizontal')

    for tick in g.get_xmajorticklabels():
        tick.set_fontname("serif")

    for tick in g.get_ymajorticklabels():
        tick.set_fontname("serif")
        
    ax_cor.patch.set_facecolor('whitesmoke')
    fig_cor.patch.set_facecolor('None')
        
    fig_cor.suptitle(Model_title, fontsize=25, x=0.55, y=0.94)
    if enable_name_list:
        fig_cor.savefig('./plots/' + save_title+'_correlations.pdf', bbox_inches='tight')
    else:
        fig_cor.savefig('./plots/' + save_title+'_correlations_no_names.pdf', bbox_inches='tight')

    
    plt.rcParams['font.size'] = temp1
    plt.rcParams['font.family'] = temp2

    return correlations_full, name_list


def runCorrelations_latent(latent_space, energy, latent_names, Model_title='ML Model', save_title='ML_model', xbins=30, ybins=30, layers=30):
    #First calculate relevant parameters from shower
    import seaborn as sns
    import pandas
    #Moments data[B,Z,X,Y]
    nMax = 20000
    latent_space = latent_space[0:nMax]
    energy = energy[0:nMax]
    
    ### GET DATAFRAMES OF LATENT SPACE and DATA
    df_latent = pandas.DataFrame(latent_space, columns=latent_names)
    df_energy = pandas.DataFrame(energy, columns=['E'])
    df = pandas.concat([df_latent,df_energy], axis=1) 
    
    ### MAKE CORRELATION PLOT
    correlations_full = df.corr()
    
    upper_triangle = np.ones((df.shape[1],df.shape[1]))
    upper_triangle[np.triu_indices(df.shape[1],1)] = 0.0
    #print(upper_triangle)
    
    correlations = correlations_full.mask(upper_triangle == 0)
    
    fig_cor = plt.figure(figsize=(10,10), dpi=200)
    ax_cor = fig_cor.add_subplot(1,1,1)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)

    temp1 = plt.rcParams['font.size']
    temp2 = plt.rcParams['font.family']
    
    plt.rcParams['font.size'] = 50 
    plt.rcParams['font.family'] = "serif"
    g = sns_plot = sns.heatmap(correlations,
        xticklabels=correlations.columns,
        yticklabels=correlations.index,
        cmap=sns.diverging_palette(250, 250, as_cmap=True),
        annot=True, ax=ax_cor, vmin=-1, vmax=1,
        annot_kws={"size": 10}, fmt=".1f", square=True, cbar=False
        )


    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 12, rotation='horizontal')
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 12, rotation='horizontal')

    for tick in g.get_xmajorticklabels():
        tick.set_fontname("serif")

    for tick in g.get_ymajorticklabels():
        tick.set_fontname("serif")
    fig_cor.suptitle(Model_title, fontsize='10')
    fig_cor.savefig('./plots/' + save_title+'_correlations.png')

    ax_cor.patch.set_facecolor('None')
    fig_cor.patch.set_facecolor('None')

    
    plt.rcParams['font.size'] = temp1
    plt.rcParams['font.family'] = temp2

    return correlations_full


def runCorrelations_latent_latent(latent_space1, latent_names1, latent_space2, latent_names2, energy, Model_title='ML Model', save_title='ML_model'):
    #First calculate relevant parameters from shower
    import seaborn as sns
    import pandas
    #Moments data[B,Z,X,Y]
    nMax = 20000
    latent_space1 = latent_space1[0:nMax]
    latent_space2 = latent_space2[0:nMax]
    energy = energy[0:nMax]
    
    ### GET DATAFRAMES OF LATENT SPACE and DATA
    df_latent1 = pandas.DataFrame(latent_space1, columns=latent_names1)
    df_latent2 = pandas.DataFrame(latent_space2, columns=latent_names2)
    df_energy = pandas.DataFrame(energy, columns=['E'])
    df = pandas.concat([df_latent1, df_energy, df_latent2, df_energy], axis=1) 
    
    ### MAKE CORRELATION PLOT   - and slice it properly for display
    correlations_full = df.corr().iloc[0 : df_latent1.shape[1]+1, df_latent1.shape[1]+1 : df.shape[1]]
    
    correlations = correlations_full
    
    fig_cor = plt.figure(figsize=(10,10))
    ax_cor = fig_cor.add_subplot(1,1,1)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)

    temp1 = plt.rcParams['font.size']
    temp2 = plt.rcParams['font.family']
    
    plt.rcParams['font.size'] = 50 
    plt.rcParams['font.family'] = "serif"
    g = sns_plot = sns.heatmap(correlations,
        xticklabels=correlations.columns,
        yticklabels=correlations.index,
        cmap=sns.diverging_palette(250, 250, as_cmap=True),
        annot=True, ax=ax_cor, vmin=-1, vmax=1,
        annot_kws={"size": 10}, fmt=".2f", square=True, cbar=False
        )


    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 12, rotation='horizontal')
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 12, rotation='horizontal')

    for tick in g.get_xmajorticklabels():
        tick.set_fontname("serif")

    for tick in g.get_ymajorticklabels():
        tick.set_fontname("serif")
    fig_cor.suptitle(Model_title, fontsize='10')
    fig_cor.savefig('./plots/' + save_title+'_correlations.png')

    ax_cor.patch.set_facecolor('None')
    fig_cor.patch.set_facecolor('None')

    
    plt.rcParams['font.size'] = temp1
    plt.rcParams['font.family'] = temp2

    return correlations_full
