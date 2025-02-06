# @file postproc.py
#  Main postprocessing driver
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT

import glob
import os
import shutil
import time

from molSimplify.Scripts.postmold import (moldpost)
from molSimplify.Scripts.postmwfn import (deloc,
                                          getcharges,
                                          getcubes,
                                          getwfnprops,
                                          globalvars,
                                          mybash)
from molSimplify.Scripts.postparse import (gampost,
                                           nbopost,
                                           terapost)


# Check if multiwfn exists
#  @param mdir Multiwfn directory
#  @return bool


def checkmultiwfn(mdir):
    if not glob.glob(mdir):
        return False
    else:
        return True

# Main postprocessing driver
#  @param rundir Runs directory
#  @param args Namespace of arguments
#  @param globs Global variables


def postproc(rundir, args, globs):
    globs = globalvars()
    # locate output files
    pdir = args.postdir if args.postdir else globs.rundir
    cmd = "find '"+pdir+"' -name *out"
    t = mybash(cmd)
    resf = t.splitlines()
    logfile = pdir+"/post.log"
    if not os.path.isdir(pdir):
        print(('\nSpecified directory '+pdir+' does not exist..\n\n'))
        return
    with open(logfile, 'a') as flog:
        flog.write('\n\n\n##### Date: ' +
                   time.strftime('%m/%d/%Y %H:%M')+'#####\n\n')
        # run summary report
        if args.pres:
            print('\nGetting runs summary..\n\n')
            flog.write('\nGetting runs summary..\n\n')
            terapost(resf, pdir, flog)
            gampost(resf, pdir, flog)
        # run nbo analysis
        if args.pnbo:
            print('\nGetting NBO summary..\n\n')
            flog.write('\nGetting NBO summary..\n\n')
            nbopost(resf, pdir, flog)
        # locate molden files
        cmd = "find "+"'"+pdir+"'"+" -name *molden"
        t = mybash(cmd)
        molf = t.splitlines()
        # parse molecular orbitals
        if args.porbinfo:
            print('\nGetting MO information..\n\n')
            flog.write('\nGetting MO information..\n\n')
            if not os.path.isdir(pdir+'/MO_files'):
                os.mkdir(pdir+'/MO_files')
            moldpost(molf, pdir, flog)
        # calculate delocalization indices
        if args.pdeloc:
            print('\nCalculating delocalization indices..\n\n')
            flog.write('\nCalculating delocalization indices..\n\n')
            if not os.path.isdir(pdir+'/Deloc_files'):
                os.mkdir(pdir+'/Deloc_files')
            deloc(molf, pdir, flog)
        # calculate charges
        if args.pcharge:
            print('\nCalculating charges..\n\n')
            flog.write('\nCalculating charges..\n\n')
            if not os.path.isdir(pdir+'/Charge_files'):
                os.mkdir(pdir+'/Charge_files')
            getcharges(molf, pdir, flog)
        # parse wavefunction
        if args.pwfninfo:
            print('\nCalculating wavefunction properties..\n\n')
            flog.write('\nCalculating wavefunction properties..\n\n')
            if not os.path.isdir(pdir+'/Wfn_files'):
                os.mkdir(pdir+'/Wfn_files')
            if not os.path.isdir(pdir+'/Cube_files'):
                os.mkdir(pdir+'/Cube_files')
            getcubes(molf, pdir, flog)
            getwfnprops(molf, pdir, flog)
            if not args.pgencubes and os.path.isdir(pdir+'/Cube_files'):
                shutil.rmtree(pdir+'/Cube_files')
        # generate cube files
        if args.pgencubes:
            print('\nGenerating cube files..\n\n')
            flog.write('\nGenerating cube files..\n\n')
            if not os.path.isdir(pdir+'/Cube_files'):
                os.mkdir(pdir+'/Cube_files')
            getcubes(molf, pdir, flog)
