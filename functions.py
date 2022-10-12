import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plt_ExampleImage(image, model_title='ML Model', save_title='ML_model', mode='comp', draw_3D=False, n=1):

    #print(image.shape)
    #data[B,Z,X,Y]
    
    cmap = mpl.cm.viridis
    cmap.set_bad('white',1.)
    
    
    #image = tf_logscale_rev_np(np.reshape(image,(1,30,30)))+1.0
    
    for k in range(n):
        figExIm = plt.figure(figsize=(6,6))
        axExIm1 = figExIm.add_subplot(1,1,1)
        image1 = np.sum(image[k], axis=0)#+1.0
        masked_array1 = np.ma.array(image1, mask=(image1==0.0))
        im1 = axExIm1.imshow(masked_array1, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                           norm=mpl.colors.LogNorm(), origin='lower')
        figExIm.patch.set_facecolor('white')
        axExIm1.title.set_text(model_title + ' {:d}'.format(k))
        axExIm1.set_xlabel('y [cells]', family='serif')
        axExIm1.set_ylabel('x [cells]', family='serif')
        figExIm.colorbar(im1)
        plt.savefig('./' + save_title+"_CollapseZ_{:d}.png".format(k))

        figExIm = plt.figure(figsize=(6,6))
        axExIm2 = figExIm.add_subplot(1,1,1)    
        image2 = np.sum(image[k], axis=1)#+1.0
        masked_array2 = np.ma.array(image2, mask=(image2==0.0))
        im2 = axExIm2.imshow(masked_array2, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                           norm=mpl.colors.LogNorm(), origin='lower') 
        figExIm.patch.set_facecolor('white')
        axExIm2.title.set_text(model_title + ' {:d}'.format(k))
        axExIm2.set_xlabel('y [cells]', family='serif')
        axExIm2.set_ylabel('z [layers]', family='serif')
        figExIm.colorbar(im2)
        plt.savefig('./' + save_title+"_CollapseX_{:d}.png".format(k))

        figExIm = plt.figure(figsize=(6,6))
        axExIm3 = figExIm.add_subplot(1,1,1)
        image3 = np.sum(image[k], axis=2)#+1.0
        masked_array3 = np.ma.array(image3, mask=(image3==0.0))
        im3 = axExIm3.imshow(masked_array3, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                           norm=mpl.colors.LogNorm(), origin='lower')
        figExIm.patch.set_facecolor('white')
        axExIm3.title.set_text(model_title + ' {:d}'.format(k))
        axExIm3.set_xlabel('x [cells]', family='serif')
        axExIm3.set_ylabel('z [layers]', family='serif')
        figExIm.colorbar(im3)
        plt.savefig('./' + save_title+"_CollapseY_{:d}.png".format(k))

    #print(np.min(image))
    

    figExIm = plt.figure(figsize=(6,6))
    axExIm1 = figExIm.add_subplot(1,1,1)
    image1 = np.mean(np.sum(image, axis=1), axis=0)#+1.0
    masked_array1 = np.ma.array(image1, mask=(image1==0.0))
    im1 = axExIm1.imshow(masked_array1, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                       norm=mpl.colors.LogNorm(), origin='lower')
    figExIm.patch.set_facecolor('white')
    axExIm1.title.set_text(model_title + 'overlay')
    axExIm1.set_xlabel('y [cells]', family='serif')
    axExIm1.set_ylabel('x [cells]', family='serif')
    figExIm.colorbar(im1)
    plt.savefig('./' + save_title+"_CollapseZSum.png")

    figExIm = plt.figure(figsize=(6,6))
    axExIm2 = figExIm.add_subplot(1,1,1)    
    image2 = np.mean(np.sum(image, axis=2), axis=0)#+1.0
    masked_array2 = np.ma.array(image2, mask=(image2==0.0))
    im2 = axExIm2.imshow(masked_array2, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                       norm=mpl.colors.LogNorm(), origin='lower') 
    figExIm.patch.set_facecolor('white')
    axExIm2.title.set_text(model_title + 'overlay')
    axExIm2.set_xlabel('y [cells]', family='serif')
    axExIm2.set_ylabel('z [layers]', family='serif')
    figExIm.colorbar(im2)
    plt.savefig('./' + save_title+"__CollapseXSum.png")
   
    figExIm = plt.figure(figsize=(6,6))
    axExIm3 = figExIm.add_subplot(1,1,1)    
    image3 = np.mean(np.sum(image, axis=3), axis=0)#+1.0
    masked_array3 = np.ma.array(image3, mask=(image3==0.0))
    im3 = axExIm3.imshow(masked_array3, filternorm=False, interpolation='none', cmap = cmap, vmin=0.001, vmax=1000,
                       norm=mpl.colors.LogNorm(), origin='lower')
    figExIm.patch.set_facecolor('white')
    axExIm3.title.set_text(model_title + 'overlay')
    axExIm3.set_xlabel('x [cells]', family='serif')
    axExIm3.set_ylabel('z [layers]', family='serif')
    figExIm.colorbar(im3)
    plt.savefig('./' + save_title+"_CollapseYSum.png")