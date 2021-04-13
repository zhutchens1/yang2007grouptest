import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

def yang_performance_test(groupid, haloid, galproperty):
    """
    For a group catalog constructed on a mock catalog, compute the galaxy-wise
    purity, completeness, and contamination based on the method of Yang et al.
    (2007, ApJ, 671:153).

    Parameters
    ---------------------------
    groupid : iterable
        Group ID numbers after applying group-finding algorithm, length = # galaxies.
    haloid : iterable
        Halo ID numbers extracted from mock catalog halos, length = # galaxies = len(groupid).
    galproperty : iterable
        Group property by which to determine the central galaxy in the group. If all values are
        >-15 and <-27, then galproperty is assumed to be a magnitude, and the central will be the
        brightest galaxy. If all values are >0, this value is assumed to be mass, and the central
        will be selected by the maximum.

    Returns
    ----------------------------
    completeness : np.array
        Completeness of group catalog. At index `i`, the number of true halo members captured by
        the group which contains galaxy `i`. Ideally, you want this to be 1 for every galaxy. The
        length matches the number of galaxies.
    contamination : np.array
        Contamination of group catalog. At index `i`, it is the fraction of interlopers (galaxies
        belonging to other halo) in the group containing galaxy `i`. Ideally, this is zero. The 
        length matches the number of galaxies.
    purity : np.array
        Purity of the group, defined as 1/(contamination fraction + completeness fraction). Ideally
        the purity is 1 for every group, describing a perfect recovery of the halos.
    """
    # prepare inputs+outputs
    groupid = np.array(groupid)
    haloid = np.array(haloid)
    galproperty = np.array(galproperty)
    assert len(haloid)==len(groupid) and len(groupid)==len(galproperty),"All input arrays must have same length."
    ngalaxies = len(groupid)
    completeness = np.full(ngalaxies, -999.)
    contamination = np.full(ngalaxies, -999.)
    purity = np.full(ngalaxies, -999.)

    # first, figure out what galproperty we have and devise a way to select centrals.
    if (galproperty>-30).all() and (galproperty<-12).all():
        central_selection = np.min # minimum mag is brightest galaxy=central
    elif (galproperty>0).all():
        central_selection = np.max # maximum mass is central

    # now iterative through individual unique group id numbers 
    unique_groups = np.unique(groupid)
    for k in unique_groups:
        # select the group
        groupsel = np.where(groupid==k)
        # get halo ID of brightest or most massive galaxy in k
        hk = haloid[groupsel][np.where(galproperty[groupsel]==central_selection(galproperty[groupsel]))]
        halosel = np.where(haloid==hk)
        # define Nt as the total number of galaxies with this halo ID
        Nt = len(haloid[halosel])
        # define Ns as the number of true halo members that are identified as part of group k
        Ns = len(groupid[np.where(groupid[halosel]==k)])
        # define Ni as the # of interlopers in the group 
        # = number of galaxies in this group that belong to other halos
        Ni = len(groupid[np.where(haloid[groupsel]!=hk)])
        # define Ng as the # of galaxies in the identified group
        Ng = len(groupid[groupsel])
        # now compute the purity,completeness,contamination
        completeness[groupsel] = Ns/Nt
        contamination[groupsel] = Ni/Nt
        purity[groupsel] = Nt/Ng
    assert (completeness!=-999.).all() and (contamination!=-999.).all() and (purity!=-999.).all()
    return completeness, contamination, purity
     



if __name__=='__main__':
    # a few tests
    print(yang_performance_test([0,0,0],[0,0,1],[-19,-18,-17]))
    # try it with a real catalog
    mock = pd.read_hdf("ECO_cat_0_Planck_memb_cat.hdf5")
    comp, cont, pur = yang_performance_test(mock.haloid, mock.haloid, mock.M_r)
    print((comp==1).all())
    print((cont==0).all())
    print((pur==1).all())
    # now do it for a bunch of them
    leadpath='/srv/two/zhutchen/dwarfonlygroups_mock/ecomocktests/proposedcentermethod/iterativecombination/halobiasmocks/'
    subdirs=['fiducial/']#,'dv0_8/','dv1_2/','central/']
    inputfilenametemplate='ECO_cat_{}_Planck_memb_cat.hdf5'

    # test purity, etc as function of halo mass in high-density mock
    mock = pd.read_hdf(leadpath+subdirs[0]+inputfilenametemplate.format(7))
    comp, conta, pur = yang_performance_test(mock.groupid, mock.haloid, mock.M_r)
    junk, uniqind = np.unique(np.array(mock.groupid), return_index=True)
    comp = comp[uniqind]
    conta = conta[uniqind]
    pur = pur[uniqind]
    halo_ngal = np.array(mock.halo_ngal)[uniqind]
    halomass = np.array(mock.loghalom)[uniqind]
    meancomp, binedges, jk = binned_statistic(halomass, comp, np.median, bins=15) 
    meancomp_n, binedges_n, jk = binned_statistic(halo_ngal, comp, np.median, bins=50) 
    perc84comp, jk, jk = binned_statistic(halomass, comp, lambda x: np.percentile(x,84.1)-np.median(x), bins=15)
    perc16comp, jk, jk = binned_statistic(halomass, comp, lambda x: np.median(x)-np.percentile(x,15.9), bins=15)
    halomasserr = np.array([perc16comp, perc84comp])
    
    fig, (ax2, ax3) = plt.subplots(ncols=1, nrows=2)
    ax2.set_title("Fiducial Mock Catalog with n = {:0.2f} (h/Mpc)$^3$".format(len(mock)/192351.36))
    ax2.plot(halomass, comp, '.', color='aqua', alpha=0.2, label='FoF Groups identified in Mock Catalog')
    #ax2.plot(binedges[:-1], meancomp, 'k^', label='Mean Completeness by Halo Mass')
    ax2.errorbar(binedges[:-1], meancomp, fmt='k^', label='Median Completeness by Halo Mass +/- 34%', yerr=halomasserr)
    ax2.set_xlabel(r"log halo mass [$h^{-1} \rm M_\odot$]")
    ax2.axvline(11.55, color='k', linestyle='solid')#, label='Gas-Richness Threshold Scale')
    ax2.axvline(12.25, color='k', linestyle='dotted')#, label='Bimodality Scale')
    ax2.set_ylabel("Completeness")
    ax2.legend(loc='best', framealpha=1)

    ax3.plot(halo_ngal, comp, '.', alpha=0.2, color='orange', label='FoF Groups identified in Mock Catalog')
    ax3.plot(binedges_n[:-1], meancomp_n, 'k^', label='Mean Completeness by Halo $N$')
    ax3.set_xlim(-1,200)
    ax3.set_xlabel("Number of True Halo Members")
    ax3.set_ylabel("Completeness")
    plt.show()

    # do it for al fiducial mocks
    avgdens, meancomp, meanconta, meanpur = [], [], [], []
    for sd in subdirs:
        for i in range(0,8):
            if i<8: # only 8 mocks
                #####################################################
                ### Read in data, prepare grpid array for output. ### 
                #####################################################
                inputfile = leadpath+sd+inputfilenametemplate.format(i)
                mock=pd.read_hdf(inputfile)
                comp, conta, pur = yang_performance_test(mock.groupid, mock.haloid, mock.M_r)
                # remap to groups
                junk, uniqind = np.unique(np.array(mock.groupid), return_index=True)
                comp = comp[uniqind]
                conta = conta[uniqind]
                pur = pur[uniqind]
                avgdens.append(len(mock)/192351.36)
                meancomp.append(np.mean(comp))
                meanconta.append(np.mean(conta))
                meanpur.append(np.mean(pur))
    
    index = np.argsort(np.array(avgdens))
    avgdens = np.array(avgdens)[index]
    meanconta = np.array(meanconta)[index]
    meancomp = np.array(meancomp)[index]
    meanpur = np.array(meanpur)[index]
    fig, (ax, ax1) = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(14,5))
    ax.axhline(1, color='k', linestyle='solid', alpha=0.4, label='Ideal Completeness of a Group')
    ax.axhline(0, color='k', linestyle='dashdot', alpha=0.4, label='Ideal Contamination of a Group')
    ax.plot(avgdens, meancomp, '-o', label='Mean Completeness of an FoF Group in Mock Catalog')
    ax.plot(avgdens, meanconta, '-o', label='Mean Contamination of an FoF Group in Mock Catalog')
    ax.legend(loc='best')
    ax1.axhline(1, color='k', linestyle='dotted', alpha=0.4, label='Ideal Purity of a Group')
    ax1.plot(avgdens, meanpur, '-o', color='green', label='Mean Purity of an FoF Group in Mock Catalog')
    ax1.set_ylim(-10,20)
    ax1.set_xlabel(r"Mock Catalog Galaxy Number Density [$h^3$ Mpc$^{-3}$]")
    ax1.legend(loc='best')
    #plt.yscale('log')
    plt.show()
