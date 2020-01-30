import numpy as np
from matplotlib import pyplot
import pandas as pd
from astropy.io import fits
from astropy import wcs
import sys
from scipy import linalg as la
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns

def T2S(freq, beam):
    """
    convert brightness temperature to Jy
    """
    c = 299792458.
    kb= 1.3806503e-23
    Jy= 1e26

    return 2 * kb * (freq*1e9/c)**2 * beam * Jy 

def SolidAngle(freq):
    
    return (0.2593*np.log(freq)**2 - 1.965*np.log(freq) + 3.865)*1e-5



def plotimage(filename):
    hdu = fits.open(filename)
    w = wcs.WCS(hdu[0].header)

    img = hdu[0].data
    img[img == 0] = np.nan
    #img[np.isnan(img)] = np.nanmedian(img)
    imgf = img[np.isfinite(img)].flatten()
    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

    vecs = np.ones((3,imgf.size))
    vecs[0,:] = x[np.isfinite(img)].flatten()#*img.flatten()
    vecs[1,:] = y[np.isfinite(img)].flatten()#*img.flatten()
    C = vecs.dot(vecs.T)
    xv = la.inv(C).dot(vecs.dot(imgf[:,np.newaxis]))

    img -= (xv[0]*x + xv[1]*y + xv[2])
        
    hduout = fits.PrimaryHDU([img], header=w.to_header())
    hdu1 = fits.HDUList([hduout])

    cmap = pyplot.get_cmap('RdBu_r')
    pyplot.figure(figsize=(12,8))
    pyplot.subplot(projection=w)
    pyplot.imshow(img,cmap=cmap,origin='lower',vmin=-0.05,vmax=0.2,aspect='auto')
    cbar = pyplot.colorbar()
    cbar.set_label('K',size=20)

    mimg = img*1
    mimg[np.isnan(img)] = 0
    mimg = gaussian_filter(mimg,sigma=2)
    pyplot.contour(mimg, cmap = pyplot.get_cmap('Greys_r'),
                   levels=[0.01,0.045,0.09,0.135,0.3,0.4])
    pyplot.gca().invert_xaxis()
    pyplot.grid()
    pyplot.xlabel(r'$\alpha$',size=20)
    pyplot.ylabel(r'$\delta$',size=20)
    #pyplot.gca().set_xlim(305,165)
    #pyplot.gca().set_ylim(130,270)


def aperphot(filename,x0, y0, freq):
    hdu = fits.open(filename)
    w = wcs.WCS(hdu[0].header)

    img = hdu[0].data
    yp, xp = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    
    ra, dec = w.wcs_pix2world(xp.flatten(),yp.flatten(),0)
    ra = np.reshape(ra, xp.shape)
    dec= np.reshape(dec,yp.shape)

    r = ((ra-x0)/np.cos(y0*np.pi/180.))**2 + (dec-y0)**2

    inner = (r < 1./60.)
    outer = (r >= 1./60.) & (r < 3./60.)

    Sin = np.nansum(img[inner])
    Sout= np.nanmean(img[outer])*np.sum(inner)
    Ssig= np.nanstd(img[outer])

    
    nu = np.linspace(26, 34)
    pixbeam = (w.wcs.cdelt[0] * np.pi/180.)**2
    S = (Sin - Sout) * T2S(freq, pixbeam)
    eS = Ssig * T2S(freq, pixbeam)
    return S, eS

if __name__ == '__main__':

    filename = sys.argv[1]
    
    catalogues = {'SNR':{'table':pd.read_csv('green.cat',delimiter=';'),
                         'x':'_RAJ2000',
                         'y':'_DEJ2000',
                         'marker':'^'},
                  'HII':{'table':pd.read_csv('hii.cat',delimiter=','),
                         'x':'RA_deg',
                         'y':'Dec_deg',
                         'marker':'s'},
                  'UCHII':{'table':pd.read_csv('uchii.cat',delimiter=','),
                           'x':'RA_deg',
                           'y':'Dec_deg',
                           'marker':'o'}  }
    plotimage(filename)
    pal = sns.color_palette("hls", 3)

    for i, (key, val) in enumerate(catalogues.items()):
        table = val['table']
        xcol = val['x']
        ycol = val['y']
        marker=val['marker']
        xy = table[[xcol,ycol]].values.astype(float)
        pyplot.gca().scatter(xy[:,0],xy[:,1],marker=marker,facecolor=(1,1,1,0),edgecolor=pal[i],
                             transform=pyplot.gca().get_transform('world'),
                             s=60)

    pyplot.xlim(385.5,-0.5)
    pyplot.ylim(60,340)
    pyplot.savefig(filename.split('.')[0]+'_sources.png')
    pyplot.clf()

    sources = [[282.3542, -0.9167, '3C193'],
               [282.04927,-01.44195,'G031.2801+00.0632']]
    for (x0,y0,name) in sources:
        S = []
        eS= []
        nu = [27,29,31,33]
        for i in range(4):
            f = 'fg6_all_SepOct2019_B{:d}.fits'.format(i)
            _S, _eS = aperphot(f, x0, y0, nu[i])
            S += [_S]
            eS += [_eS]

        pyplot.errorbar(nu,S,fmt='o',yerr=eS, capsize=3)
        pyplot.xlabel('Frequency (GHz)')
        pyplot.ylabel('S(Jy)')
        pyplot.title(name)
        pyplot.grid()
        pyplot.savefig(name+'_spectrum.png')

        pyplot.clf()

