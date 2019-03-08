''' Code for updating Bad Pixel Masks from a set of definitions to another
Function-driven instead of Class, for simpler paralellisnm
'''

import os
import socket
import sys
import time
import logging
import argparse
import copy
import uuid
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
try:
    import matplotlib.pyplot as plt
except:
    pass
import fitsio

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Global variables
BITDEF_INI = None

def open_fits(fnm):
    ''' Open the FITS, read the data and store it on a list
    '''
    tmp = fitsio.FITS(fnm)
    data, header = [], []
    for extension in tmp:
        ext_tmp = np.copy(extension.read())
        data.append(ext_tmp)
        header_tmp = copy.deepcopy(extension.read_header())
        header.append(header_tmp)
    tmp.close()
    if (len(data) == 0):
        logging.error('No extensions were found on {0}'.format(fnm))
        exit(1)
    elif (len(data) == 1):
        return data[0], header[0]
    elif (len(data) >= 1):
        return data, header

def load_bitdef(d1=None):
    ''' Load tables of bit definitions and pass them to a dictionary
    '''
    global BITDEF_INI
    try:
        kw = {
            'sep' : None,
            'comment' : '#', 
            'names' : ['def', 'bit'], 
            'engine' : 'python',
        }
        t1 = pd.read_table(d1, **kw)
        t2 = pd.read_table(d2, **kw)
    except:
        t_i = 'pandas{0} doesn\'t support guess sep'.format(pd.__version__) 
        logging.info(t_i)
        kw.update({'sep' : '\s+',})
        t1 = pd.read_table(d1, **kw)
    # Construct the dictionaries
    BITDEF_INI = dict(zip(t1['def'], t1['bit']))
    # Change data type to unsigned integers, to match the dtype of the BPMs
    for k in BITDEF_INI:
        BITDEF_INI[k] = np.uint(BITDEF_INI[k])
    return True 

def load_msk01(fnm):
    ''' Method to load a mask from a text file, containing just 1's and 0's
    Values 1: mask it, 0: leave untouched
    Parameters
    ----------
    fnm: str
        Filename of the text file mask
    Returns
    -------
    m: ndarray
        2D array containing the mask
    '''
    m = np.loadtxt(fnm)
    return m

def bit_count(int_type):
    ''' Function to count the amount of bits composing a number, that is the
    number of base 2 components on which it can be separated, and then
    reconstructed by simply sum them. Idea from Wiki Python.
    Brian Kernighan's way for counting bits. Thismethod was in fact
    discovered by Wegner and then Lehmer in the 60s
    This method counts bit-wise. Each iteration is not simply a step of 1.
    Example: iter1: (2066, 2065), iter2: (2064, 2063), iter3: (2048, 2047)
    Inputs
    - int_type: integer
    Output
    - integer with the number of base-2 numbers needed for the decomposition
    '''
    counter = 0
    while int_type:
        int_type &= int_type - 1
        counter += 1
    return counter

def bit_decompose(int_x):
    ''' Function to decompose a number in base-2 numbers. This is performed by
    two binary operators. Idea from Stackoverflow.
        x << y
    Returns x with the bits shifted to the left by y places (and new bits on
    the right-hand-side are zeros). This is the same as multiplying x by 2**y.
        x & y
    Does a "bitwise and". Each bit of the output is 1 if the corresponding bit
    of x AND of y is 1, otherwise it's 0.
    Inputs
    - int_x: integer
    Returns
    - list of base-2 values from which adding them, the input integer can be
    recovered
    '''
    base2 = []
    i = 1
    while (i <= int_x):
        if (i & int_x):
            base2.append(i)
        i <<= 1
    return base2

def flatten_list(list_2levels):
    ''' Function to flatten a list of lists, generating an output with
    all elements in a single level. No duplicate drop neither sort are
    performed
    '''
    f = lambda x: [item for sublist in x for item in sublist]
    res = f(list_2levels)
    return res

def split_bitmask_FITS(arr, ccdnum, save_fits=False, outnm=None):
    ''' Return a n-dimensional array were each layer is an individual bitmask,
    then, can be loaded into DS9. This function helps as diagnostic.
    '''
    # I need 3 lists/arrays to perform the splitting:
    # 1) different values composing the array
    # 2) the bits on which the above values can be decomposed
    # 3) the number of unique bits used on the above decomposition
    # First get all the different values the mask has
    diff_val = np.sort(np.unique(arr.ravel()))
    # Decompose each of the unique values in its bits
    decomp2bit = []
    for d_i in diff_val:
        dcomp = bit_decompose(d_i)
        decomp2bit.append(dcomp)
    diff_val = tuple(diff_val)
    decomp2bit = tuple(decomp2bit)
    # Be careful to keep diff_val and decomp2bit with the same element order,
    # because both will be compared
    # Get all the used bits, by the unique components of the flatten list
    # Option for collection of unordered unique elements:
    #   bit_uniq = repr(sorted(set(bit_uniq)))
    bit_uniq = flatten_list(list(decomp2bit))
    bit_uniq = np.sort(np.unique(bit_uniq))
    # Safe checks
    #
    # Check there are no missing definitions!
    #
    # Go through the unique bits, and get the positions of the values
    # containig such bit
    bit2val = dict()
    # Over the unique bits
    for b in bit_uniq:
        tmp_b = []
        # Over the bits composing each of the values of the array
        for idx_nb, nb in enumerate(decomp2bit):
            if (b in nb):
                tmp_b.append(diff_val[idx_nb])
        # Fill a dictionary with the values that contains every bit
        bit2val.update({'BIT_{0}'.format(b) : tmp_b})
    # Create a ndimensional matrix were to store the positions having
    # each of the bits. As many layers as unique bits are required to
    # construct all the values
    is1st = True
    for ib in bit_uniq:
        # Zeros or NaN?
        # tmp_arr = np.full(arr.shape, np.nan)
        tmp_arr = np.zeros_like(arr)
        # Where do the initial array contains the values that are 
        # composed by the actual bit?
        for k in bit2val['BIT_{0}'.format(ib)]:
            tmp_arr[np.where(arr == k)] = ib
            # print 'bit: ', ib, ' N: ', len(np.where(arr == k)[0])
        # print '==========', ib, len(np.where(tmp_arr == ib)[0])
        # Add new axis
        tmp_arr = tmp_arr[np.newaxis , : , :]
        if is1st:
            ndimBit = tmp_arr
            is1st = False
        # NOTE: FITS files recognizes depth as the 1st dimesion
        # ndimBit = np.dstack((ndimBit, tmp_arr))
        ndimBit = np.vstack((ndimBit, tmp_arr))
    if save_fits:
        if (outnm is None):
            outnm = str(uuid.uuid4())
            outnm = os.path.joint(outnm, '.fits')
        if os.path.exists(outnm):
            t_w = 'File {0} exists. Will not overwrite'.format(outnm)
            logging.warning(t_w)
        else: 
            fits = fitsio.FITS(outnm, 'rw')
            fits.write(ndimBit)
            fits[-1].write_checksum()
            hlist = [
                {'name' : 'CCDNUM', 'value' : ccdnum, 'comment' : 'CCD number'},
                {'name' : 'COMMENT', 
                 'value' : 'Multilayer bitmask, Francisco Paz-Chinchon'}
                ]
            fits[-1].write_keys(hlist)
            fits.close()
            t_i = 'Multilayer bitmaks saved {0}'.format(outnm)
            logging.info(t_i)
    return True

def get_args():
    ''' Construct the argument parser 
    '''
    t_gral = 'Code to add a BIT to a region, on a existing set of files.'
    t_epi = 'BPM format is assumed to be DES-wise'
    argu = argparse.ArgumentParser(description=t_gral, epilog=t_epi)
    # input table of definitions
    h0 = 'Filename for the set of definitions for the bitmask. Format: 2'
    h0 += ' columns, with bit name in the first column, and'
    h0 += ' bit integer in the second'
    argu.add_argument('--ini', help=h0, metavar='filename')
    h2 = 'List of files to be masked with a new region'
    argu.add_argument('--bpm', help=h2, metavar='filename')
    h3 = 'Filename of the template mask for the region to be masked. Use a'
    h3 += ' plain text file'
    argu.add_argument('--reg', help=h3, metavar='filename')
    h4 = 'Bit to be added to the target region'
    argu.add_argument('--bit', help=h4, metavar='bit', type=int)
    h5 = 'Prefix to be used for naming output files.'
    h5 += ' Default is to add PID to the filename. If prefix is'
    h5 += ' given, then the output will be \'{prefix}_c{ccdnum}.fits\''
    argu.add_argument('--prefix', '-p', help=h5, metavar='str')
    h6 = 'Number of processors to run in parallel. Default: N-1 cpu'
    argu.add_argument('--nproc', '-n', help=h6, metavar='integer', type=int)
    #
    argu = argu.parse_args()
    return argu

def aux_main():
    logging.info(socket.gethostname())
    # Argument parser
    argu = get_args()
    NPROC = mp.cpu_count() - 1
    BIT = argu.bit
    if (argu.prefix is None):
        prefix = str(os.getpid())
    if (argu.nproc is not None):
        NPROC = argu.nproc
    t_i = 'Running {0} processes in parallel'.format(NPROC)
    logging.info(t_i)
    # Load bit definition
    if (argu.ini is not None):
        load_bitdef(d1=argu.ini)
        aux_dic = dict(zip(BITDEF_INI.values(), BITDEF_INI.keys()))
        if (BIT in BITDEF_INI.values()):
            t_i = '{1}={0}'.format(BIT, aux_dic[BIT])
            t_i += ' will be added to the masked region'
            logging.info(t_i)
        else:
            t_w = 'Bit {0} not contained on bit definition'.format(BIT)
            logging.warning(t_w)
    else:
        t_w = 'No bit definition was input to corroborate BIT existence'
        logging.warning(t_w)
    # Load mask of region
    msk = load_msk01(argu.reg).astype(bool)
    # Open BPMs, store objects in a list
    aux_fnm_bpm = np.genfromtxt(argu.bpm, dtype='str')
    bpm = []
    outfnm = []
    for f in aux_fnm_bpm:
        data, header = open_fits(f)
        bpm.append((data, header))
        if isinstance(data, list):
            for d in data:
                if (d.shape != msk.shape):
                    logging.error('Mask and data doesn\'t match dimensions')
                    exit(1)
        else:
            if (data.shape != msk.shape):
                logging.error('Mask and data doesn\'t match dimensions')
                exit(1)
        # Store  filename for the output
        outfnm.append(prefix + '_' + os.path.basename(f))
    # Identify the bits in that region, based in unique values
    for idx, (x, hdr) in enumerate(bpm):
        # Masked region
        sub = np.ma.masked_where(~msk, x)
        # Unique values
        uniq = np.unique(sub.compressed())
        # Iterate unique values, adding the BIT
        for u in uniq:
            bit_cnt = bit_count(u)
            bit_uni = bit_decompose(u)
            if (bit_cnt == 0):
                # Case when pixel value is zero
                sub[np.where(sub == u)] += BIT
            elif (BIT not in bit_uni):
                # Case when BIT is not contained on the pixel value
                sub[np.where(sub == u)] += BIT
            else:
                # Case when the BIT is already on the pixel value
                pass
        #
        # it = np.nditer(sub.compressed(), flags=['multi_index'])
        # while not it.finished:
        #     print(it[0], it.multi_index)
        #
        # Write out FITS file
        if True: #try:
            fits = fitsio.FITS(outfnm[idx], 'rw')
            fits.write(sub.data, header=hdr)
            txt = 'fpazch updated region mask.'
            hlist = [{'name' : 'comment', 'value' : txt},]
            fits[-1].write_keys(hlist)
            fits.close()
            t_i = 'FITS file written: {0}'.format(outfnm[idx])
            logging.info(t_i)
        if False: #except:
            t_e = sys.exc_info()[0]
            logging.error(t_e)
    return True


if __name__ == '__main__':

    aux_main()

