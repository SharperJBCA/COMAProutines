import numpy as np
from matplotlib import pyplot
from astropy.io import fits
from astropy import wcs
import sys
from scipy import linalg as la
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patches import Circle
if __name__ == '__main__':
    
    
    hdu = fits.open(sys.argv[1])
    w = wcs.WCS(hdu[0].header)

    img = hdu[2].data
    img[img == 0] = np.nan
    #img[np.isnan(img)] = np.nanmedian(img)
    imgf = img[np.isfinite(img)].flatten()
    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

    vecs = np.ones((7,imgf.size))
    vecs[0,:] = x[np.isfinite(img)].flatten()#*img.flatten()
    vecs[1,:] = y[np.isfinite(img)].flatten()#*img.flatten()
    vecs[2,:] = x[np.isfinite(img)].flatten()**2#*img.flatten()
    vecs[3,:] = y[np.isfinite(img)].flatten()**2#*img.flatten()
    vecs[4,:] = x[np.isfinite(img)].flatten()**3#*img.flatten()
    vecs[5,:] = y[np.isfinite(img)].flatten()**3#*img.flatten()

    C = vecs.dot(vecs.T)
    xv = la.inv(C).dot(vecs.dot(imgf[:,np.newaxis]))

    img -= (xv[0]*x + xv[1]*y +\
            xv[2]*x**2 + xv[3]*y**2 +\
            xv[4]*x**3 + xv[5]*y**3 + xv[6])
        
    hduout = fits.PrimaryHDU([img], header=w.to_header())
    hdu1 = fits.HDUList([hduout])
    #hdu1.writeto('test_'+sys.argv[1],overwrite=True)
    #img -= np.nanmedian(img,axis=0)[np.newaxis,:]
    #img -= np.nanmedian(img,axis=1)[:,np.newaxis]

    cmap = pyplot.get_cmap('RdBu_r')
    pyplot.figure(figsize=(12,8))
    pyplot.subplot(projection=w)

    x0,y0 = w.all_world2pix(283.2,-1.9322,0)
    
    xpos, ypos = np.meshgrid(np.arange(img.shape[1],dtype=int),
                             np.arange(img.shape[0],dtype=int))
    r0 = 30./60. / w.wcs.cdelt[0]
    circle = Circle((283.2,-1.9322),30/60., edgecolor='red', facecolor='none', linewidth=2,
                    transform=pyplot.gca().get_transform('fk5'))
    rpos = ((xpos-x0)**2 + (ypos-y0)**2)**0.5
    select = np.where((rpos.flatten() < r0))[0]
    img_flat = img.flatten()
    map_rms = np.nanstd(img_flat[select])
    map_mean=np.nanmean(img_flat[select])

    print('Stats:', map_rms, map_mean)
    #img_flat[select] = 1000
    #img = np.reshape(img_flat, img.shape)

    pyplot.imshow(img,cmap=cmap,origin='lower',vmin=-0.05,vmax=0.05,aspect='auto')
    cbar = pyplot.colorbar()
    cbar.set_label('K',size=20)

    mimg = img*1
    mimg[np.isnan(img)] = 0
    mimg = gaussian_filter(mimg,sigma=2)
    #pyplot.contour(mimg, cmap = pyplot.get_cmap('Greys_r'),
    #               levels=[-0.02,-0.015,-0.01,-0.005,0,0.01,0.02,0.045,0.07,0.08,0.09,0.135,0.3,0.4])
    pyplot.gca().invert_xaxis()
    pyplot.grid()
    pyplot.xlabel(r'$\alpha$',size=20)
    pyplot.ylabel(r'$\delta$',size=20)
    pyplot.gca().set_xlim(0.8*img.shape[1],0.3*img.shape[1])
    pyplot.gca().set_ylim(0.3*img.shape[0],0.7*img.shape[0])
    pyplot.gca().add_patch(circle)
    pyplot.savefig('Band0_W43_SepOct19_Naive.png')
    pyplot.show()
