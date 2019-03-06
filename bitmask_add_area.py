''' Code for updating Bad Pixel Masks from a set of definitions to another
Function-driven instead of Class, for simpler paralellisnm
'''

import os
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
        t2 = pd.read_table(d2, **kw)
    # Construct the dictionaries
    BITDEF_INI = dict(zip(t1['def'], t1['bit']))
    # Change data type to unsigned integers, to match the dtype of the BPMs
    for k in BITDEF_INI:
        BITDEF_INI[k] = np.uint(BITDEF_INI[k])
    return True 

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

# ===========================================================
# ===========================================================
# ===========================================================

def diff_bitdef(show_diff=False):
    ''' Compare both dictionaries bit definitions. Also compare the bit value
    for the same bit definition, on both schemas. Leave it as general as 
    possible
    Returns
    - 2 lists, one containing the bit definition names not being common for 
    both the definitions, and the other containing the bit definitions for 
    the bits having different name in both sets. Both lists are strings,
    not integer, because are bitdef, not the bits numerical values
    '''
    # Set intersection and differences. Beware of the use of set() and 
    # its capabilities/limitations
    inter = set(BITDEF_INI.keys()).intersection(set(BITDEF_END.keys()))
    diff = set(BITDEF_INI.keys()).symmetric_difference(set(BITDEF_END.keys()))
    # Compare in terms of keys, for both sets.
    if show_diff:
        for k_diff in diff:
            # Do not discard the possibility of both sets having unique bit
            # definitions
            if (k_diff in set(BITDEF_INI.keys())):
                t_i0 = 'Different BITS: shown in INITIAL but not in FINAL'
                t_i0 += ' bitdef {0}:{1}'.format(k_diff, BITDEF_INI[k_diff])
                logging.info(t_i0)
            if (k_diff in set(BITDEF_END.keys())):
                t_i1 = 'Different BITS: shown in FINAL but not in INITIAL'
                t_i1 += ' bitdef {0}:{1}'.format(k_diff, BITDEF_END[k_diff])
                logging.info(t_i1)
    # Compare in terms of values, for same key across both sets.
    diff_keyvalues = []
    for k_inter in inter:
        if (BITDEF_INI[k_inter] != BITDEF_END[k_inter]):
            diff_keyvalues.append(k_iter)
            if show_diff:
                t_i2 = 'Different definitions: '
                t_i2 += '{0}:{1} INITIAL'.format(k_inter, BITDEF_INI[k_inter])
                t_i2 += ' {0}:{1} FINAL'.format(k_inter, BITDEF_END[k_inter])
                logging.info(t_i2)
    return list(diff), diff_keyvalues



def change_bitmask(in_list):
    # For parralelism
    arr, header, tab_ini, tab_end, compare_tab = in_list
    ccdnum = int(header['CCDNUM'])
    #
    t_i = 'Working on CCD={0}'.format(ccdnum)
    logging.info(t_i)
    # Previous information: get the list of bits composing the whole CCD
    aux_unique = np.unique(arr)
    ccd_bits = map(bit_decompose, aux_unique[1:])
    ccd_bits = flatten_list(ccd_bits)
    ccd_bits = list(set(ccd_bits))
    t_i = 'Bits composing the CCD {0}: {1}'.format(ccdnum, sorted(ccd_bits))
    logging.info(t_i)
    if not( set(ccd_bits).issubset(set(BITDEF_INI.values())) ):
        t_e = 'Not all bits composing the CCD are contained on the initial'
        t_e += ' bit definition. Exiting'
        logging.error(t_e)
        exit(1)
    # 1) Load sections corresponding to each bit.
    s_ini = pd.read_table(
        tab_ini, comment='#', sep='\s+', 
        names=['ccdnum', 'x0', 'x1', 'y0', 'y1', 'bpmdef_ini']
    )
    s_end = pd.read_table(
        tab_end, comment='#', sep='\s+',
        names=['ccdnum', 'x0', 'x1', 'y0', 'y1', 'bpmdef_end']
    ) 
    # Optional
    # 1.1) Comparison between the tables defining the polygons to be masked,
    # for the current CCD
    if compare_tab:
        # Redirects the print to a file
        orig_stdout = sys.stdout
        comp_fnm = 'compare_bad_pixels_lists_PID{0}.txt'.format(os.getpid())
        f = open(comp_fnm, 'w')
        sys.stdout = f
        # As on the above methods, crossmatch the tables defining sections
        df_merge = pd.merge(s_ini, s_end, on=['x0', 'x1', 'y0', 'y1'])
        if (len(df_merge.index) != len(s_ini.index)): 
            print 'Different number of matches!!!'
        print df_merge
        print 'pre Y4\n', s_ini
        print 'post Y4\n', s_end
        print '=' * 80
        sys.stdout = orig_stdout
        f.close()
    # 2) Check which definitions are not common for both sets. These should be
    # the bits to be updated. Check also which definitions having the same
    # key (name) have different numerical value.
    diff_bitname, diff_keyval = diff_bitdef()
    # Check if there are insconsitencies between the two set of definitions, 
    # for the bits in common
    if (len(diff_keyval) > 0):
        t_e = 'Bit value for the same bit definition differs. Each bitdef'
        t_e += ' should have the same value, regardless the set. Bits'
        t_e += ' showing this issue are: {0}'.format(','.join(diff_keyval))
        t_e += ' Exiting'
        logging.error(t_e)
        exit(1)
    # 3) Per CCD, translate the bitmask. Only add the new BITS, replacing
    # the old ones
    # Iterate CCD by CCD. Use new set to put new bits in place
    #
    # 3.1) Refine sections to only those in the current CCD
    s_ini = s_ini.loc[s_ini['ccdnum'] == ccdnum]
    s_ini.reset_index(drop=True, inplace=True)
    s_end = s_end.loc[s_end['ccdnum'] == ccdnum]
    s_end.reset_index(drop=True, inplace=True)
    # Check there are masks to be updated in the current CCD.
    if ((len(s_ini.index) == 0) or (len(s_end.index) == 0)):
        t_w = 'There are no masks to be updated for the current CCD.'
        t_w += ' Returning initial CCD array'
        logging.warning(t_w)
        return tuple([header, arr])
    # 3.2) Crossmatch the bits to be updated. We already checked the other 
    # bits have common definition in both sets
    bits2update = [BITDEF_END[s] for s in diff_bitname] 
    s_end = s_end.loc[s_end['bpmdef_end'].isin(bits2update)]
    # Important: the following assumes that both set of polygons (old and new)
    # have the same polygon defined by coordinates, but with different bits. 
    # Then, the update will be done on this specific set of NEW bits 
    # coming from the crsoomathc of spatial match of coordinates
    # Crossmatch both definitions based on the vertices of polygons
    s_update = pd.merge(s_ini, s_end, on=['x0', 'x1', 'y0', 'y1'])
    # Check the crossmatch is not missing any row
    if (len(s_update.index) != len(s_end.index)):
        t_e = 'The crossmatch between old and new definition has missing'
        t_e += ' common entries. Exiting'
        logging.error(t_e)
        exit(1)
    # 3.3) Go through the different polygons coming out of the crossmatch
    # and check if the old-bit is in place, in which case the new-bit will
    # take its place
    # NOTE: Remember table coordinates starts in 1. Shape is (4096, 2048)
    # Iterate over polygons, adding the new-bit
    for idx, r1 in s_update.iterrows():
        y0, y1, x0, x1 = r1['y0'], r1['y1'], r1['x0'], r1['x1']
        oldbit = r1['bpmdef_ini']
        newbit = r1['bpmdef_end']
        xsub = arr[y0 - 1 : y1 , x0 - 1 : x1] 
        # Iterate over the pixels in each section
        for index, px in np.ndenumerate(xsub):
            # Bits composing the pixel
            px_bits = bit_decompose(px)  
            # Check the old bit is among the pixel bits. If not, an error
            # must be raised and exit. The exit is drastic, but we need the
            # old-bit to be replaced
            if not (oldbit in px_bits):
                t_e = 'No old-bit={0} in section of'.format(oldbit)
                t_e += ' coords=[{0},{1},{2},{3}]'.format(x0, x1, y0, y1)
                t_e += ' CCD {0}.'.format(ccdnum)
                t_e += ' New-bit={0},'.format(newbit)
                t_e += ' pixel-bits={0},'.format(px_bits)
                logging.error(t_e)
                t_e = 'Writting out the original array'
                logging.error(t_e)
                return tuple([header, arr])
        # At this point, old-bit is in ALL the pixel-bits
        # Add the new-bit to the section, but still don't remove the old, 
        # because we need it to be present for overlapping sections.  
        xsub += newbit
        # Replace the section into the CCD-array
        arr[y0 - 1 : y1, x0 - 1 : x1] = xsub
        #
        if False:
            tmp = map(bit_decompose, np.unique(xsub))
            tmp = flatten_list(tmp)
            tmp = list(set(tmp))
            tmp = sorted(tmp)
            print 'Added new-bit, before remove old-bit: {0}'.format(tmp)
    # Now must iterate again, over the same sections, and remove the old-bit
    # From the above loop, we already know old-bit is present in the 
    # sections
    # Over sections
    for idx, r2 in s_update.iterrows():
        y0, y1, x0, x1 = r2['y0'], r2['y1'], r2['x0'], r2['x1']
        oldbit_x = r2['bpmdef_ini']
        newbit_x = r2['bpmdef_end']
        xsub_x = arr[y0 - 1 : y1 , x0 - 1 : x1] 
        # Over pixels in section
        for x in np.nditer(xsub_x, op_flags=['readwrite']):
            x_bits = bit_decompose(x)  
            if (oldbit_x in x_bits):
                x -= oldbit_x
        # Replace section into the CCD-array
        arr[y0 - 1 : y1, x0 - 1 : x1] = xsub_x
        #
        if False:
            tmp = map(bit_decompose, np.unique(xsub_x))
            tmp = flatten_list(tmp)
            tmp = list(set(tmp))
            tmp = sorted(tmp)
            print 'Added new-bit, after remove old-bit: {0}'.format(tmp)
    # 4) Finally, return the updated array
    return tuple([header, arr])

def operate_bitmask(fnm_list=None, tab_ini=None, tab_end=None, 
                    ext=0, nproc=4, prefix=None, fix_dtype=True):
    ''' Load set of bitmask from the input list. This function needs to change
    while the code advances/matures
    '''
    # Bits to be updated are those who are in the new set but not in the old
    diff_bitname, diff_keyval = diff_bitdef(show_diff=True)
    t_i = 'Bitdef to be updated: {0}'.format(diff_bitname)
    logging.info(t_i)
    if (len(diff_keyval) > 0):
        t_i = 'Bits having different keys between sets:'
        t_i += ' {0}'.format(diff_keyval)
        logging.info(t_i)
    # Load the set of full paths for BPM files to be translated
    df_fnm = pd.read_table(fnm_list, names=['bpm'])
    parallel_list = []
    # Columns for Dataframe to be used as auxiliary for naming
    tmp_col1 = []
    tmp_col2 = []
    for ind, f in df_fnm.iterrows():
        x, hdr = open_fits(f['bpm'])
        # We expect the FITS file to have only one extension
        if ((len(x) > 1) or (len(hdr) > 1)):
            t_w = 'FITS file {0} has more than 1 extension'.format(f['bpm'])
            logging.warning(t_w)
        # Get data from FITS
        x = x[ext]
        hdr = hdr[ext]
        # Some BPMs were written with a bug: even when data is integer,
        # the datatype was not. I need to fix it before do comparison
        # against other integers (masks)
        if fix_dtype:
            x = x.astype('uint16') 
        # Aux for naming
        tmp_col1.append(int(hdr['CCDNUM']))
        tmp_col2.append(os.path.basename(f['bpm']))
        # Fill the auxiliary list for parallel call
        parallel_list.append([x, hdr, tab_ini, tab_end, False])
    # Dataframe for naming
    df_aux = pd.DataFrame({'ccdnum' : tmp_col1, 'filename' : tmp_col2})
    # Running in parallel
    P1 = mp.Pool(processes=nproc)   
    xnew = P1.map(change_bitmask, parallel_list)
    P1.close()
    # Write out the modified bitmasks 
    for data in xnew:
        ccdnum = int(data[0]['CCDNUM'])
        fi_aux = df_aux.loc[df_aux['ccdnum'] == ccdnum, 'filename'].values[0]
        if (prefix is None):
            outnm = 'updated_' + fi_aux
        else:
            outnm = prefix + '_c{0:02}.fits'.format(ccdnum)
        try:
            fits = fitsio.FITS(outnm, 'rw')
            fits.write(data[1], header=data[0])
            txt = 'fpazch updated bit definitions.'
            txt += ' Original file {0}'.format(fi_aux)
            hlist = [{'name' : 'comment', 'value' : txt},]
            fits[-1].write_keys(hlist)
            fits.close()
            t_i = 'FITS file written: {0}'.format(outnm)
        except:
            t_e = sys.exc_info()[0]
            logging.error(t_e)
    #
    if False:
        # Construct a filename to FITS to harbor one array per each of the 
        # bits used in the masking. Example: is a mask has 5 bits, then the
        # bit-layer file will be composed by 5 arrays each one with a
        # different unique bit
        out_bitLayer = os.path.join(os.getcwd(), 
                                    'bitLayer_' + os.path.basename(f['bpm']))
        # Split bitmask into its components
        # NOTE: would be helpful to have this function run with the final
        # translated BPM and also with the previous
        split_bitmask_FITS(x_new, ccdnum, save_fits=True, outnm=out_bitLayer)
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
    h5 += ' Default is to add \'updated\' to the filename. If prefix is'
    h5 += ' given, then the output will be \'{prefix}_c{ccdnum}.fits\''
    argu.add_argument('--prefix', '-p', help=h5, metavar='str')
    h6 = 'Number of processors to run in parallel. Default: N-1 cpu'
    argu.add_argument('--nproc', '-n', help=h6, metavar='integer', type=int)
    #
    argu = argu.parse_args()
    return argu

# ===========================================================
# ===========================================================
# ===========================================================

def aux_main():
    logging.info(socket.hostname)
    # Argument parser
    argu = get_args()
    NPROC = mp.cpu_count() - 1
    if (argu.nproc is not None):
        NPROC = argu.nproc
    t_i = 'Running {0} processes in parallel'.format(NPROC)
    logging.info(t_i)
    # Load bit definition
    load_bitdef(d1=argu.ini)
    # Open BPMs, store objects in a list
    aux_fnm_bpm = np.genfromtxt(argu.bpm ,dtype='str')
    bpm = []
    for f in aux_fnm_bpm:
        data, header = open_fits(f)
        bpm.append((data, header))
    # Load mask of region


    # Identify the bits in that region

    # Add the new bits to the region, pixel by pixel. Avoid add the same
    # bit twice


    # Write out FITS file





if __name__ == '__main__':
    t0 = time.time()
    NPROC = mp.cpu_count() - 1
    # Argument parser
    argu = get_args()
    if (argu.nproc is not None):
        NPROC = argu.nproc
    t_i = 'Running {0} processes in parallel'.format(NPROC)
    logging.info(t_i)
    # Load the tables, storing info in global variables
    load_bitdef(d1=argu.ini, d2=argu.end)
    # Load BPM FITS file and split in its components
    operate_bitmask(fnm_list=argu.mig, tab_ini=argu.polyA, 
                    tab_end=argu.polyB, nproc=NPROC, prefix=argu.prefix)
    # 
    t1 = time.time()
    t_i = 'Elapsed time for this run: {0:.2f} min'.format((t1 - t0) / 60.)
    logging.info(t_i)
