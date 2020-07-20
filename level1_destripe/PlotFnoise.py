import numpy as np
from matplotlib import pyplot
import h5py
import healpy as hp
import sys
from tqdm import tqdm
from comancpipeline.Tools import Coordinates
from matplotlib.transforms import ScaledTranslation
from scipy.signal import fftconvolve
import seaborn as sns

def angular_seperation(theta1, theta2, phi1, phi2):
    
    d2r = np.pi/180.

    A = np.sin(phi1*d2r)*np.sin(phi2*d2r)
    B = np.cos(phi1*d2r)*np.cos(phi2*d2r)*np.cos((theta2-theta1)*d2r)
    return np.arccos(A+B)/d2r

def MAD(d,axis=0):

    med_d = np.nanmedian(d,axis=axis)
    rms = np.sqrt(np.nanmedian((d-med_d)**2,axis=axis))*1.48

    return rms
    

def fnoise_list(mode = 'GFields'):
    filelist = np.loadtxt(sys.argv[1],dtype=str)
    obsid = np.array([int(f.split('-')[1]) for f in filelist])
    filelist = filelist[np.argsort(obsid)]
    obsid = np.sort(obsid)

    fnoise = np.zeros(filelist.size)
    enoise = np.zeros(filelist.size)
    feed = None
    ifeed = 0
    isfg4 = np.zeros(filelist.size,dtype=bool)
    dist = np.zeros(filelist.size)
    pyplot.figure(figsize=(20,5))
    for ifile, filename in enumerate(filelist):
        
        print(filename)
        try:
            data = h5py.File(filename,'r')
        except OSError:
            print('{} cannot be opened (Resource unavailable)'.format(filename))
            fnoise[ifile] = np.nan

        if mode.lower() in data['level1/comap'].attrs['source'].decode('utf-8').lower():
            isfg4[ifile] = True

        try:
            fits = data['level2/fnoise_fits'][ifeed,-1,1:-2,:]
            fnoise[ifile] = np.median(fits[:,1])
            enoise[ifile] = np.sqrt(np.median(np.abs(fits[:,1]-fnoise[ifile])**2))*1.4826
        except:
            print('{} not processed'.format(filename.split('/')[-1]))
            fnoise[ifile] = np.nan

        if isinstance(feed, type(None)):
            feed = data['level1/spectrometer/feeds'][ifeed]

        # Calculate sun distance
        mjd = data['level1/spectrometer/MJD'][0:1]
        lon=-118.2941
        lat=37.2314
        ra_sun, dec_sun, raddist = Coordinates.getPlanetPosition('SUN', lon, lat, mjd)
        az_sun, el_sun = Coordinates.e2h(ra_sun, dec_sun, mjd, lon, lat)
        ra  = data['level1/spectrometer/pixel_pointing/pixel_ra'][0,0:1]
        dec = data['level1/spectrometer/pixel_pointing/pixel_dec'][0,0:1]
        dist[ifile] = el_sun[0]#angular_seperation(ra_sun, ra, dec_sun, dec)
        data.close()
    good = (fnoise > -1.2) & (fnoise < -0.5) & np.isfinite(fnoise) & (fnoise != -1)
    with open('Plots/{}_good.list'.format(mode),'w') as f:
        for line in filelist[good]:
            f.write('{}\n'.format(line))

    pyplot.errorbar(np.arange(fnoise.size),fnoise,fmt='.',yerr=enoise,capsize=3)
    pyplot.errorbar(np.arange(fnoise.size)[good],fnoise[good],fmt='.',yerr=enoise[good],capsize=3)

    pyplot.xticks(np.arange(fnoise.size),obsid, rotation=90,size=8)
    pyplot.ylim(-2,-0.8)
    pyplot.grid()
    pyplot.savefig('Plots/Fnoise_feed{}_{}.png'.format(feed,mode),bbox_inches='tight')
    pyplot.savefig('Plots/Fnoise_feed{}_{}.pdf'.format(feed,mode),bbox_inches='tight')
    pyplot.show()

def fnoise_plots(mode,ifeed):
    filelist = np.loadtxt(sys.argv[1],dtype=str)
    obsid = np.array([int(f.split('-')[1]) for f in filelist])
    filelist = filelist[np.argsort(obsid)]
    obsid = np.sort(obsid)

    fnoise = np.zeros(filelist.size)
    enoise = np.zeros(filelist.size)
    feed = None
    isfg4 = np.zeros(filelist.size,dtype=bool)
    dist = np.zeros(filelist.size)

    fnoise_power = np.zeros((filelist.size,64*4))
    alphas = np.zeros((filelist.size,64*4))

    for ifile, filename in enumerate(filelist):
        
        try:
            data = h5py.File(filename,'r')
        except OSError:
            print('{} cannot be opened (Resource unavailable)'.format(filename))
            fnoise[ifile] = np.nan

        if mode.lower() in data['level1/comap'].attrs['source'].decode('utf-8').lower():
            isfg4[ifile] = True

        try:
            fits = data['level2/fnoise_fits'][ifeed,:,:,:]
            fnoise[ifile] = np.median(fits[:,1])
            enoise[ifile] = np.sqrt(np.median(np.abs(fits[:,1]-fnoise[ifile])**2))*1.4826
            ps = data['level2/powerspectra'][ifeed,:,:,:]
            rms = data['level2/wnoise_auto'][ifeed,:,:,:]
            nu = data['level2/freqspectra'][ifeed,:,:,:]
            freq = data['level1/spectrometer/frequency'][...]
            bw = 16
            freq = np.mean(np.reshape(freq, (freq.shape[0],freq.shape[1]//bw, bw)),axis=-1).flatten()
            sfreq = np.argsort(freq)
        
            fnoise_power[ifile,:] = (rms[:,:,0]**2 * (1/fits[:,:,0])**fits[:,:,1]).flatten()[sfreq]
            alphas[ifile,:] = (fits[:,:,1]).flatten()[sfreq]

            #print(nu.shape,ps.shape, rms.shape, fits.shape)
            #pyplot.plot(freq[sfreq],fnoise_power[ifile,:])
        except IOError:
            print('{} not processed'.format(filename.split('/')[-1]))
            fnoise[ifile] = np.nan

        if isinstance(feed, type(None)):
            feed = data['level1/spectrometer/feeds'][ifeed]

        # Calculate sun distance
        mjd = data['level1/spectrometer/MJD'][0:1]
        lon=-118.2941
        lat=37.2314
        ra_sun, dec_sun, raddist = Coordinates.getPlanetPosition('SUN', lon, lat, mjd)
        az_sun, el_sun = Coordinates.e2h(ra_sun, dec_sun, mjd, lon, lat)
        ra = data['level1/spectrometer/pixel_pointing/pixel_ra'][0,0:1]
        dec = data['level1/spectrometer/pixel_pointing/pixel_dec'][0,0:1]
        dist[ifile] = el_sun[0]#angular_seperation(ra_sun, ra, dec_sun, dec)
        data.close()


    # Plot obs ID vs fnoise power
    pyplot.imshow(np.log10(fnoise_power*1e3),aspect='auto',origin='lower',
                  extent=[np.min(freq),np.max(freq),-.5,fnoise_power.shape[0]-0.5])
    pyplot.yticks(np.arange(fnoise_power.shape[0])-0.5, obsid, rotation=0,
                  ha='right',va='center',size=10)
    ax = pyplot.gca()
    fig = pyplot.gcf()
    offset = ScaledTranslation(-0.08,0.02,fig.transFigure)
    for label in ax.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    pyplot.grid()
    pyplot.xlabel('Frequency (GHz)')
    pyplot.ylabel('obs ID')
    pyplot.colorbar(label=r'$\mathrm{log}_{10}$(mK)')
    pyplot.title('Feed {}'.format(feed))
    pyplot.savefig('Plots/fnoise_gfields_Feed{}.png'.format(feed),bbox_inches='tight')
    pyplot.clf()
    # Plot obs ID vs fnoise power
    pyplot.imshow(alphas,aspect='auto',origin='lower',vmin=-1.5,vmax=-0.9,
                  extent=[np.min(freq),np.max(freq),-.5,fnoise_power.shape[0]-0.5])
    pyplot.yticks(np.arange(fnoise_power.shape[0])-0.5, obsid, rotation=0,
                  ha='right',va='center',size=10)
    ax = pyplot.gca()
    fig = pyplot.gcf()
    for label in ax.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    pyplot.grid()
    pyplot.xlabel('Frequency (GHz)')
    pyplot.ylabel('obs ID')
    pyplot.colorbar(label=r'$\alpha$')
    pyplot.title('Feed {}'.format(feed))
    pyplot.savefig('Plots/alphas_gfields_Feed{}.png'.format(feed),bbox_inches='tight')
    pyplot.clf()


def fnoise_matrix():
    filelist = np.loadtxt(sys.argv[1],dtype=str)
    obsid = np.array([int(f.split('-')[1]) for f in filelist])
    filelist = filelist[np.argsort(obsid)]
    obsid = np.sort(obsid)

    nfeeds, nbands, nchans = 18, 4, 64
    fnoise = np.zeros((nfeeds,nbands,nchans))# nfeeds*nbands*nchans))
    counts = np.zeros((nfeeds,nbands,nchans))# nfeeds*nbands*nchans))
    alphas = np.zeros((nfeeds,nbands,nchans))# nfeeds*nbands*nchans))
    alphaN = np.zeros((nfeeds,nbands,nchans))# nfeeds*nbands*nchans))

    feeds = None
    freqs = None
    f0 = 1
    for ifile, filename in enumerate(tqdm(filelist)):
        
        try:
            data = h5py.File(filename,'r')
        except OSError:
            print('{} cannot be opened (Resource unavailable)'.format(filename))

        try:
            fits = data['level2/fnoise_fits'][...,:]
            rms = data['level2/wnoise_auto'][...,0]

            power = rms**2 * (f0/10**fits[...,0])**fits[...,1]
            good = np.isfinite(power)
            fnoise[good] += power[good]
            counts[good] += 1
            alphas[good] += fits[good,1]
            alphaN[good] += 1


        except:
            print('{} not processed'.format(filename.split('/')[-1]))

        if isinstance(feeds, type(None)):
            feeds = data['level1/spectrometer/feeds'][:]
            freqs = data['level1/spectrometer/frequency'][...]
            freqs = np.mean(np.reshape(freqs,(freqs.shape[0], freqs.shape[1]//16,16)),axis=-1)
            print(freqs.shape, fnoise.shape)
        data.close()
    fnoise = fnoise/counts
    alphas = alphas/counts
    fnoise[fnoise == 0] = np.nan
    alphas[alphas == 0] = np.nan
    axes = [pyplot.subplot(1,2,1+i) for i in range(2)]

    from scipy.signal import medfilt
    for i in range(fnoise.shape[0]):
        if feeds[i] == 20:
            continue

        pyplot.sca(axes[int(feeds[i]//10)])
        nu, d, alpha = freqs.flatten(),fnoise[i].flatten(),alphas[i].flatten()
        nusort = np.argsort(nu)
        nu, d, alpha = nu[nusort],d[nusort], alpha[nusort]
        good = (np.abs(d-medfilt(d,15)) < 1e-4)# & ( (alpha-medfilt(alpha,15)) < -0.05 )
        pyplot.plot(nu[good],alpha[good],'-',label=feeds[i])
    for ax in axes:
        pyplot.sca(ax)
        pyplot.grid()
        pyplot.legend(prop={'size':8})
        pyplot.ylabel(r'$\alpha$')
        pyplot.xlabel('Frequency (GHz)')
    pyplot.tight_layout()
    pyplot.clf()

    import copy
    channelmask = np.zeros((alphas.shape[0],alphas.shape[1],alphas.shape[2])).astype(bool)
    from scipy.signal import medfilt
    for i in range(fnoise.shape[0]):
        if feeds[i] == 20:
            continue

        nu, d, alpha = freqs.flatten(),fnoise[i].flatten(),alphas[i].flatten()
        nusort = np.argsort(nu)
        nu, d, alpha = nu[nusort],d[nusort], alpha[nusort]
        good = (np.abs(d-medfilt(d,15)) < 1e-4)

        fraction = np.sum(good)/len(good)

        gd2 = copy.copy(good)
        gd2[:] = False
        gd2[nusort] = good
        channelmask[i] = np.reshape(gd2,channelmask[i].shape)

        pyplot.plot(feeds[i],fraction,'.k')#,label=feeds[i])
    pyplot.grid()
    pyplot.ylabel(r'Good Fraction')
    pyplot.xlabel('Feed')
    pyplot.xticks(feeds)
    pyplot.tight_layout()
    pyplot.clf()

    np.save('Plots/channelmask.npy',channelmask)


def wnoise_matrix():
    filelist = np.loadtxt(sys.argv[1],dtype=str)
    obsid = np.array([int(f.split('-')[1]) for f in filelist])
    filelist = filelist[np.argsort(obsid)]
    obsid = np.sort(obsid)

    nfeeds, nbands, nchans = 18, 4, 64
    wnoise = np.zeros((nfeeds,nbands,nchans))# nfeeds*nbands*nchans))
    counts = np.zeros((nfeeds,nbands,nchans))# nfeeds*nbands*nchans))

    feeds = None
    freqs = None
    f0 = 1
    for ifile, filename in enumerate(tqdm(filelist)):
        
        try:
            data = h5py.File(filename,'r')
        except OSError:
            print('{} cannot be opened (Resource unavailable)'.format(filename))

        try:
            rms = data['level2/wnoise_auto'][...,0]
            power = rms
            gd = np.isfinite(rms)
            wnoise[gd] += power[gd]#[good]
            counts[gd] += 1
            

        except:
            print('{} not processed'.format(filename.split('/')[-1]))

        if isinstance(feeds, type(None)):
            feeds = data['level1/spectrometer/feeds'][:]
            freqs = data['level1/spectrometer/frequency'][...]
            freqs = np.mean(np.reshape(freqs,(freqs.shape[0], freqs.shape[1]//16,16)),axis=-1)
        data.close()
    wnoise = wnoise/counts
    wnoise[wnoise == 0] = np.nan

    pyplot.imshow(wnoise[:,0,:])
    pyplot.show()

    channelmask = np.zeros((wnoise.shape[0],wnoise.shape[1],wnoise.shape[2])).astype(bool)
    from scipy.signal import medfilt
    BW = 2e9/1024 * 16.
    for i in range(wnoise.shape[0]):
        if feeds[i] == 20:
            continue

        wn = wnoise[i]
        lims = [5,5,2,2]
        for j in range(wn.shape[0]):
            resid = (wn[j]-np.nanmin(wn[j])) * np.sqrt(BW/50)
            #pyplot.plot( resid)
            gd = np.isfinite(resid)
            pmdl = np.poly1d(np.polyfit(np.arange(resid.size)[gd],resid[gd],2))
            resid -= pmdl(np.arange(resid.size))
            resid -= np.nanmedian(resid)
            resid = np.abs(resid)

            bad = (resid > lims[j]).astype(float)
            bad = fftconvolve(bad, np.ones(3)/3, mode='same')
            bad = (bad > 0.1)
            channelmask[i,j,:] = resid > 5 
            #pyplot.show()
            channelmask[i,j,54:] = True


        #pyplot.plot(channelmask[i].flatten())
    np.save('Plots/channelmask.npy',channelmask)
   # pyplot.show()
    #     good = (np.abs(d-medfilt(d,15)) < 1e-4)

    #     fraction = np.sum(good)/len(good)

    #     gd2 = copy.copy(good)
    #     gd2[:] = False
    #     gd2[nusort] = good
    #     channelmask[i] = np.reshape(gd2,channelmask[i].shape)

    #     pyplot.plot(feeds[i],fraction,'.k')#,label=feeds[i])
    # pyplot.grid()
    # pyplot.ylabel(r'Good Fraction')
    # pyplot.xlabel('Feed')
    # pyplot.xticks(feeds)
    # pyplot.tight_layout()
    # pyplot.clf()


def fnoise_summary(mode='GFields'):
    filelist = np.loadtxt(sys.argv[1],dtype=str)
    obsid = np.array([int(f.split('-')[1]) for f in filelist])
    filelist = filelist[np.argsort(obsid)]
    obsid = np.sort(obsid)

    nfeeds, nbands, nchans = 18, 4, 64
    fnoise = np.zeros((len(filelist),nfeeds,nbands,nchans))*np.nan# nfeeds*nbands*nchans))
    counts = np.zeros((len(filelist),nfeeds,nbands,nchans))*np.nan# nfeeds*nbands*nchans))
    alphas = np.zeros((len(filelist),nfeeds,nbands,nchans))*np.nan# nfeeds*nbands*nchans))
    alphaN = np.zeros((len(filelist),nfeeds,nbands,nchans))*np.nan# nfeeds*nbands*nchans))
    wnoise = np.zeros((len(filelist),nfeeds,nbands,nchans))*np.nan# nfeeds*nbands*nchans))

    feeds = None
    freqs = None
    f0 = 1

    def fnoisefunc(P,x,rms):
        return rms**2 * (1+(x/10**P[0])**P[1])
    
    for ifile, filename in enumerate(tqdm(filelist)):
        
        try:
            data = h5py.File(filename,'r')
        except OSError:
            print('{} cannot be opened (Resource unavailable)'.format(filename))

            
        try:
            fits = data['level2/fnoise_fits'][...,:]
            rms = data['level2/wnoise_auto'][...,0]

            ps = data['level2/powerspectra'][...]
            nu = data['level2/freqspectra'][...]
            bands = data['level1/spectrometer/bands'][:]
            feeds = data['level1/spectrometer/feeds'][:]
            obsid = data.filename.split('-')[1]

            palette = sns.color_palette('husl',8)
            for ifeed in range(len(feeds)):
                if feeds[ifeed] == 20:
                    continue
                # for band in range(4):
                #     pyplot.subplot(2,2,1+band)
                #     for channel in range(64):
                #         pyplot.plot(nu[ifeed,band,channel,:],ps[ifeed,band,channel,:]*1e6,
                #                     color=palette[0],alpha=0.5,linewidth=2,zorder=0)
                #         pyplot.plot(nu[0,band,channel,:],fnoisefunc(fits[ifeed,band,channel,:],
                #                                                     nu[0,band,channel,:],
                #                                                     rms[ifeed,band,channel])*1e6,
                #                     color='k',alpha=0.5,zorder=1)
                #     pyplot.yscale('log')
                #     pyplot.xscale('log')
                #     pyplot.title(bands[band].decode('utf-8'),size=10)
                #     pyplot.xlim(np.min(nu[0,band,channel,:]), np.max(nu[0,band,channel,:]))
                #     pyplot.xticks(size=8)
                #     pyplot.yticks(size=8)
                #     pyplot.xlabel('Hz',size=10)
                #     pyplot.ylabel('Power (mK^2)',size=10)
                #     pyplot.grid()
                # pyplot.tight_layout()
                # pyplot.suptitle('Obsid {} Feed {:02d}'.format(obsid,feeds[ifeed]))
                # pyplot.savefig('Plots/PowerSpecs/{}_{:02d}_ps.png'.format(obsid,feeds[ifeed]),bbox_inches='tight')
                # pyplot.clf()
            
            
            power = rms**2 * (f0/10**fits[...,0])**fits[...,1]
            good = np.isfinite(power)
            fnoise[ifile,good] = power[good]
            counts[ifile,good] = 1
            alphas[ifile,good] = fits[good,1]
            alphaN[ifile,good] = 1
            wnoise[ifile,good] = rms[good]**2

            
        except KeyError:
            print('{} not processed'.format(filename.split('/')[-1]))

        # if isinstance(feeds, type(None)):
        feeds = data['level1/spectrometer/feeds'][:]
        freqs = data['level1/spectrometer/frequency'][...]
        #pyplot.plot(freqs.flatten())
        nfreqs=freqs.size
        freqs = np.mean(np.reshape(freqs,(freqs.shape[0], freqs.shape[1]//16,16)),axis=-1)
        data.close()
    # fnoise = np.sqrt(fnoise/counts)
    # alphas = alphas/counts
    # fnoise[fnoise == 0] = np.nan
    # alphas[alphas == 0] = np.nan
    # wnoise = np.sqrt(wnoise/counts)
    # wnoise[wnoise == 0] = np.nan


    pyplot.figure(figsize=(16,16))

    fnoise = np.sqrt(fnoise)
    wnoise = np.sqrt(wnoise)

    freqs = freqs.flatten()
    for ifeed, feed in enumerate(feeds):
        if feed == 20:
            continue

        sfreqs = np.argsort(freqs)

        fn = fnoise[:,ifeed,...]
        fn_mean = np.nanmedian(fn,axis=0).flatten()[sfreqs]*1e3
        fn_rms  = MAD(fn,axis=0).flatten()[sfreqs]*1e3

        wn_mean = np.nanmean(wnoise[:,ifeed,...],axis=0).flatten()[sfreqs]*1e3
        wn_rms  = MAD(wnoise[:,ifeed,...],axis=0).flatten()[sfreqs]*1e3

        pyplot.subplot(3,3,1+np.mod(ifeed,9))
        plotinfo, = pyplot.plot(freqs[sfreqs],fn_mean,label='1/f at 1Hz')
        print(fn_mean[:5])
        print(fn_rms[:5])
        pyplot.fill_between(freqs[sfreqs], fn_mean-fn_rms, fn_mean+fn_rms, color=plotinfo.get_color(),alpha=0.25,zorder=0)
        plotinfo, = pyplot.plot(freqs[sfreqs],wn_mean,label='white noise')
        pyplot.fill_between(freqs[sfreqs], wn_mean-wn_rms, wn_mean+wn_rms, color=plotinfo.get_color(),alpha=0.25,zorder=0)

        pyplot.ylabel('RMS (mK)',size=10)
        pyplot.xlabel('Frequency (GHz)',size=10)
        pyplot.title('Feed {}'.format(feed),size=10)
        pyplot.xticks(size=8)
        pyplot.yticks(size=8)
        pyplot.grid()
        pyplot.yscale('log')
        pyplot.legend(prop={'size':7})
        for tick in pyplot.gca().yaxis.get_minor_ticks():
            tick.label.set_fontsize(8)
        # print('FEED {}'.format(feed),
        #       'fnoise power',
        #       np.nanmedian(fnoise[ifeed]),np.nanstd(fnoise[ifeed]),
        #       'wnoise power',
        #       np.nanmedian(wnoise[ifeed]),np.nanstd(wnoise[ifeed]),
        #       'fnoise alpha',
        #       np.nanmedian(alphas[ifeed]),np.nanstd(alphas[ifeed]))
        if (np.mod(ifeed,9) == 8) | (ifeed == len(feeds)-1):            
            pyplot.tight_layout()
            pyplot.savefig('Plots/{}_{}_FnoiseSummary.png'.format(mode,ifeed),bbox_inches='tight')
            pyplot.clf()

    pyplot.savefig('Plots/{}_{}_FnoiseSummary.png'.format(mode,ifeed),bbox_inches='tight')
    pyplot.clf()



    for ifeed, feed in enumerate(feeds):
        if feed == 20:
            continue
        al_mean = np.nanmedian(alphas[:,ifeed,...],axis=0).flatten()[sfreqs]
        al_rms  = MAD(alphas[:,ifeed,...],axis=0).flatten()[sfreqs]

        pyplot.subplot(3,3,1+np.mod(ifeed,9))
        sfreqs = np.argsort(freqs)
        plotinfo, = pyplot.plot(freqs[sfreqs],al_mean,label=r'1/f $\alpha$')
        pyplot.fill_between(freqs[sfreqs], al_mean-al_rms, al_mean+al_rms, color=plotinfo.get_color(),alpha=0.25,zorder=0)

        pyplot.ylabel(r'$\alpha$',size=10)
        pyplot.xlabel('Frequency (GHz)',size=10)
        pyplot.title('Feed {}'.format(feed),size=10)
        pyplot.xticks(size=8)
        pyplot.yticks(size=8)
        pyplot.grid()
       # pyplot.yscale('log')
        pyplot.legend(prop={'size':7})
        for tick in pyplot.gca().yaxis.get_minor_ticks():
            tick.label.set_fontsize(8)
        # print('FEED {}'.format(feed),
        #       'fnoise power',
        #       np.nanmedian(fnoise[ifeed]),np.nanstd(fnoise[ifeed]),
        #       'wnoise power',
        #       np.nanmedian(wnoise[ifeed]),np.nanstd(wnoise[ifeed]),
        #       'fnoise alpha',
        #       np.nanmedian(alphas[ifeed]),np.nanstd(alphas[ifeed]))
        if (np.mod(ifeed,9) == 8) | (ifeed == len(feeds)-1):            
            pyplot.tight_layout()
            pyplot.savefig('Plots/{}_{}_FnoiseSummary_alpha.png'.format(mode,ifeed),bbox_inches='tight')
            pyplot.clf()

    pyplot.savefig('Plots/{}_{}_FnoiseSummary_alpha.png'.format(mode,ifeed),bbox_inches='tight')
    pyplot.clf()




if __name__ == "__main__":
    mode =sys.argv[1].split('/')[-1].split('_')[0]
    #wnoise_matrix()
    fnoise_list(mode)
    fnoise_summary(mode)
    #for ifeed in tqdm(range(17)):
    #    fnoise_plots(mode,ifeed)
    #fnoise_summary(mode)#(mode='fg7')
    #wnoise_matrix()
