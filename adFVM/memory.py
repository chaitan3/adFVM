from __future__ import print_function
import os

_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    return _VmB('VmSize:') - since


def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since


def stacksize(since=0.0):
    '''Return stack size in bytes.
    '''
    return _VmB('VmStk:') - since

import resource
from . import parallel, config
def printMemUsage():
    return
    if not config.user.profile_mem:
        return
    mem_max = resource.getrusage(resource.RUSAGE_SELF)[2]*resource.getpagesize()/(1024*1024.)
    mem_curr = resident()/(1024*1024)
    print('Max memory usage (process {0}):'.format(parallel.rank), mem_max, 'MB')
    print('Current memory usage (process {0}):'.format(parallel.rank), mem_curr, 'MB')
    #import guppy 
    #print guppy.hpy().heap()
