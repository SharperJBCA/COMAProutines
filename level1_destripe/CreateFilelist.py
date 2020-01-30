import glob
import h5py

import sys
if __name__=="__main__":


    datadir = sys.argv[1]
    target = sys.argv[2]

    filelist = glob.glob('{}/*.hd5'.format(datadir))
    if len(filelist) == 0:
        print('No files found')
    else:
        for f in filelist:
            try:
                d = h5py.File(f,'r')
            except OSError:
                continue
            try:
                src = d['comap'].attrs['source'].decode('ASCII')
            except KeyError:
                continue
            if target in src:
                print(f)
            d.close()
