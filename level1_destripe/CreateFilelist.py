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
                src = d['level1/comap'].attrs['source'].decode('ASCII')
                comment = d['level1/comap'].attrs['comment'].decode('utf-8')
            except KeyError:
                continue
            if (target in src) & (not 'Sky nod' in comment):
                print(f)
            d.close()
