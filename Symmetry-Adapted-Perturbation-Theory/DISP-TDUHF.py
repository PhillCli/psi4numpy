# FIXME: Add proper citations
"""
A Psi4 input script to compute second order dispersion out of TDUHF response. As a note this is, by far,
not the most efficiently algorithm, but certainly the most verbose.
References:
- TDHF equations and algorithms taken from [Amos:1985:2186] and [Helgaker:2000]
- Gauss-Legendre integration from [Amos:1985:2186] and [Jiemchooroj:2006:124306]
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith", "Filip Brzek"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-11-14"

import time
import numpy as np
import scipy.linalg as la
np.set_printoptions(precision=6, linewidth=200, threshold=2000, suppress=True)
import psi4
from helper_RPA import *


# Set memory & output file
psi4.set_memory('2048 MB')
psi4.core.set_output_file('output.dat', False)
psi4.core.set_num_threads(2)

# Set molecule to dimer (xyz coordinates, default angstroms)
molecule = "N2"

#basis = "3-21G"
#basis = "6-31G"
basis = "aug-cc-pvqz"

geometries = {
# closed shell
              "ArHF" : """
Ar 0.000 0.000 -3.470
--
H  0.000 0.000  0.000
F  0.000 0.000  0.920
""",
              "He2" : """
He  0.000 0.000 -1.48169297
--
He  0.000 0.000  1.48169297
""",
# open shell
              "N2" : """
0 4
N  0.000 0.000  0.550
--
0 4
N  0.000 0.000 -0.550
""",
              "LiH" : """
0 2
Li 0.000 0.000  0.3982
--
0 2
H  0.000 0.000 -1.1945
"""}

geo_str = geometries.get(molecule)
dimer = psi4.geometry(geo_str)


psi4.set_options({"basis" : basis,
                  "scf_type": "df",
                  #"scf_type": "direct",
                  "reference": "uhf",
                  #"save_jk": True,
                  "guess": "sad",
                  "e_convergence": 1e-13,
                  "d_convergence": 1e-13})

dimer.reset_point_group('c1')
dimer.fix_orientation(True)
dimer.fix_com(True)
dimer.update_geometry()

# sanity on monomers
nfrags = dimer.nfragments()
if nfrags != 2:
    psi4.core.clean()
    raise Exception("Found {} fragments, must be 2.".format(nfrags))

# mon geometries
monomerA = dimer.extract_subsets(1, 2)
monomerA.set_name("MonomerA")

monomerB = dimer.extract_subsets(2, 1)
monomerB.set_name("MonomerB")

# FIXME: Add handling if nocc for UHF is zero, e.g. H
t = time.time()

scf_e_A, wfnA = psi4.energy('SCF', return_wfn=True, molecule=monomerA)

scf_e_B, wfnB = psi4.energy('SCF', return_wfn=True, molecule=monomerB)

print('SCF took                %5.3f seconds' % ( time.time() - t))

print("")
print("[DEBUG]: E_A: {} E_B: {}".format(scf_e_A, scf_e_B))

# AO-MO coeffs in DCBS
# Cx_Y_z := x - occupied or virtual
#           Y - monomere A or B
#           z - a(b) for electron alpha(beta)

# monA
Co_A_a = wfnA.Ca_subset("AO", "OCC")
Cv_A_a = wfnA.Ca_subset("AO", "VIR")
eps_A_a = np.asarray(wfnA.epsilon_a())

Co_A_b = wfnA.Cb_subset("AO", "OCC")
Cv_A_b = wfnA.Cb_subset("AO", "VIR")
eps_A_b = np.asarray(wfnA.epsilon_b())

# monB
Co_B_a = wfnB.Ca_subset("AO", "OCC")
Cv_B_a = wfnB.Ca_subset("AO", "VIR")
eps_B_a = np.asarray(wfnB.epsilon_a())

Co_B_b = wfnB.Cb_subset("AO", "OCC")
Cv_B_b = wfnB.Cb_subset("AO", "VIR")
eps_B_b = np.asarray(wfnB.epsilon_b())

# sanity check
assert(wfnA.nmo() == wfnB.nmo())
nbf = wfnA.nmo()

# no occupied orbitals
nocc_A_a = wfnA.nalpha()
nocc_A_b = wfnA.nbeta()

nocc_B_a = wfnB.nalpha()
nocc_B_b = wfnB.nbeta()

# no virtual orbitals
nvir_A_a = nbf - nocc_A_a
nvir_A_b = nbf - nocc_A_b

nvir_B_a = nbf - nocc_B_a
nvir_B_b = nbf - nocc_B_b

# occ*virt block sizes
# no alpha->beta excitations
nov_A_a = nocc_A_a * nvir_A_a
nov_A_b = nocc_A_b * nvir_A_b

nov_B_a = nocc_B_a * nvir_B_a
nov_B_b = nocc_B_b * nvir_B_b

print("Mon: A")
print("Nocc  (alpha):  {:4d}\tNocc  (beta):  {:4d}  (o)".format(nocc_A_a,
                                                                nocc_A_b))
print("Nvir  (alpha):  {:4d}\tNvir  (beta):  {:4d}  (v)".format(nvir_A_a,
                                                                nvir_A_b))
print("Nrot  (alpha):  {:4d}\tNrot  (beta):  {:4d}  (ov)".format(nov_A_a,
                                                                 nov_A_b))
print("")
print("Mon: B")
print("Nocc  (alpha):  {:4d}\tNocc  (beta):  {:4d}  (o)".format(nocc_B_a,
                                                                nocc_B_b))
print("Nvir  (alpha):  {:4d}\tNvir  (beta):  {:4d}  (v)".format(nvir_B_a,
                                                                nvir_B_b))
print("Nrot  (alpha):  {:4d}\tNrot  (beta):  {:4d}  (ov)".format(nov_B_a,
                                                                 nov_B_b))
print("")

# orb energies
# monA
eps_v_A_a = eps_A_a[nocc_A_a:]
eps_v_A_b = eps_A_b[nocc_A_b:]

eps_o_A_a = eps_A_a[:nocc_A_a]
eps_o_A_b = eps_A_b[:nocc_A_b]

# monB
eps_v_B_a = eps_B_a[nocc_B_a:]
eps_v_B_b = eps_B_b[nocc_B_b:]

eps_o_B_a = eps_B_a[:nocc_B_a]
eps_o_B_b = eps_B_b[:nocc_B_b]

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfnA.basisset())
S = np.asarray(mints.ao_overlap())
I = mints.ao_eri()
# ERI transformation
# NOTE: SAPT notation in the comments, CC(?) notation in the var name (TD-HF.py) taken from Psi4NumPy
# NOTE: SAPT a(b) -> occupied of monA(monB); r(s) -> virtual of monA(monB)

########
# monA #
########
# v_oovv
# (i, a): alpha; (j, b): alpha
v_ijab_A_aa = np.asarray(mints.mo_transform(I, Co_A_a, Co_A_a, Cv_A_a, Cv_A_a))
# (i, a): beta; (j, b): beta
v_ijab_A_bb = np.asarray(mints.mo_transform(I, Co_A_b, Co_A_b, Cv_A_b, Cv_A_b))

print("[DEBUG] monA: ERI(ijab) (alpha, alpha) shape: {}".format(v_ijab_A_aa.shape))
print("[DEBUG] monA: ERI(ijab) (beta,  beta)  shape: {}".format(v_ijab_A_bb.shape))

# v_ovov
# (i, a): alpha; (j, b): alpha
v_iajb_A_aa = np.asarray(mints.mo_transform(I, Co_A_a, Cv_A_a, Co_A_a, Cv_A_a))
# (i, a): beta; (j, b): beta
v_iajb_A_bb = np.asarray(mints.mo_transform(I, Co_A_b, Cv_A_b, Co_A_b, Cv_A_b))
# (i, a): alpha; (j, b): beta
v_iajb_A_ab = np.asarray(mints.mo_transform(I, Co_A_a, Cv_A_a, Co_A_b, Cv_A_b))
# (i, a): beta; (j, b): alpha
v_iajb_A_ba = np.asarray(mints.mo_transform(I, Co_A_b, Cv_A_b, Co_A_a, Cv_A_a))
print("[DEBUG] monA: ERI(iajb) (alpha, alpha) shape: {}".format(v_iajb_A_aa.shape))
print("[DEBUG] monA: ERI(iajb) (beta,  beta)  shape: {}".format(v_iajb_A_bb.shape))
print("[DEBUG] monA: ERI(iajb) (alpha, beta)  shape: {}".format(v_iajb_A_ab.shape))
print("[DEBUG] monA: ERI(iajb) (beta,  alpha) shape: {}".format(v_iajb_A_ba.shape))

########
# monB #
########
# v_oovv
# (i, a): alpha; (j, b): alpha
v_ijab_B_aa = np.asarray(mints.mo_transform(I, Co_B_a, Co_B_a, Cv_B_a, Cv_B_a))
# (i, a): beta; (j, b): beta
v_ijab_B_bb = np.asarray(mints.mo_transform(I, Co_B_b, Co_B_b, Cv_B_b, Cv_B_b))

print("[DEBUG] monB: ERI(ijab) (alpha, alpha) shape: {}".format(v_ijab_B_aa.shape))
print("[DEBUG] monB: ERI(ijab) (beta,  beta)  shape: {}".format(v_ijab_B_bb.shape))

# v_ovov
# (i, a): alpha; (j, b): alpha
v_iajb_B_aa = np.asarray(mints.mo_transform(I, Co_B_a, Cv_B_a, Co_B_a, Cv_B_a))
# (i, a): beta; (j, b): beta
v_iajb_B_bb = np.asarray(mints.mo_transform(I, Co_B_b, Cv_B_b, Co_B_b, Cv_B_b))
# (i, a): alpha; (j, b): beta
v_iajb_B_ab = np.asarray(mints.mo_transform(I, Co_B_a, Cv_B_a, Co_B_b, Cv_B_b))
# (i, a): beta; (j, b): alpha
v_iajb_B_ba = np.asarray(mints.mo_transform(I, Co_B_b, Cv_B_b, Co_B_a, Cv_B_a))

print("[DEBUG] monB: ERI(iajb) (alpha, alpha) shape: {}".format(v_iajb_B_aa.shape))
print("[DEBUG] monB: ERI(iajb) (beta,  beta)  shape: {}".format(v_iajb_B_bb.shape))
print("[DEBUG] monB: ERI(iajb) (alpha, beta)  shape: {}".format(v_iajb_B_ab.shape))
print("[DEBUG] monB: ERI(iajb) (beta,  alpha) shape: {}".format(v_iajb_B_ba.shape))

#########
# dimer #
#########
#   AABB
# v_ovov
# (i, a): alpha; (j, b): alpha
v_AB_aa = np.asarray(mints.mo_transform(I, Co_A_a, Cv_A_a, Co_B_a, Cv_B_a))
# (i, a): beta; (j, b): beta
v_AB_bb = np.asarray(mints.mo_transform(I, Co_A_b, Cv_A_b, Co_B_b, Cv_B_b))
# (i, a): alpha; (j, b): beta
v_AB_ab = np.asarray(mints.mo_transform(I, Co_A_a, Cv_A_a, Co_B_b, Cv_B_b))
# (i, a): beta; (j, b): alpha
v_AB_ba = np.asarray(mints.mo_transform(I, Co_A_b, Cv_A_b, Co_B_a, Cv_B_a))

print("[DEBUG] dimer: ERI(iajb) (alpha, alpha) shape: {}".format(v_AB_aa.shape))
print("[DEBUG] dimer: ERI(iajb) (beta, beta) shape: {}".format(v_AB_bb.shape))
print("[DEBUG] dimer: ERI(iajb) (alpha, beta) shape: {}".format(v_AB_ab.shape))
print("[DEBUG] dimer: ERI(iajb) (beta, alpha) shape: {}".format(v_AB_ba.shape))

Co_A_a = np.asarray(Co_A_a)
Co_A_b = np.asarray(Co_A_b)
Cv_A_a = np.asarray(Cv_A_a)
Cv_A_b = np.asarray(Cv_A_b)

Co_B_a = np.asarray(Co_B_a)
Co_B_b = np.asarray(Co_B_b)
Cv_B_a = np.asarray(Cv_B_a)
Cv_B_b = np.asarray(Cv_B_b)
print('Integral transform took %5.3f seconds\n' % ( time.time() - t))

print("Building hessians for monomers ...\n")
print("MonA: ")
eig_A, Z_A = tduhf_eigen(
    ((nocc_A_a, nocc_A_b), (nvir_A_a, nvir_A_b),),
    ((eps_o_A_a, eps_o_A_b), (eps_v_A_a, eps_v_A_b)),
    ((v_ijab_A_aa, v_ijab_A_bb),
     (v_iajb_A_aa, v_iajb_A_bb, v_iajb_A_ab, v_iajb_A_ba)),
     (nov_A_a, nov_A_b),
    coupled=False)

print("MonB: ")
eig_B, Z_B = tduhf_eigen(
    ((nocc_B_a, nocc_B_b), (nvir_B_a, nvir_B_b),),
    ((eps_o_B_a, eps_o_B_b), (eps_v_B_a, eps_v_B_b)),
    ((v_ijab_B_aa, v_ijab_B_bb),
     (v_iajb_B_aa, v_iajb_B_bb, v_iajb_B_ab, v_iajb_B_ba)),
     (nov_B_a, nov_B_b),
    coupled=False)

print("Calculating dispersion energy ...")
# UHF version same as eq. 37-39 (only a->r, b->s excitations)
# "Dispersion interaction of high-spin open-shell complexes in the random phase approximation"
# JCP, 119, 10497 (2003), doi: 10.1063/1.1620496

# reshape dimer ERIs
v_AB_aa.shape = (nov_A_a, nov_B_a)
v_AB_bb.shape = (nov_A_b, nov_B_b)
v_AB_ab.shape = (nov_A_a, nov_B_b)
v_AB_ba.shape = (nov_A_b, nov_B_a)

v1 = np.hstack((v_AB_aa, v_AB_ab))
v2 = np.hstack((v_AB_ba, v_AB_bb))
v_AB = np.vstack((v1, v2))

t = time.time()

# Hapka&Zuchowski SAPT(DFT) 2013
E = np.dot(Z_A.real.T, v_AB)
D = np.dot(E, Z_B.real)

# helper for denominator reshaping
shape = lambda dim: (-1,) + tuple([1]*(dim -1))

omegas_pq = 1/((eig_A.real**0.5).reshape(shape(2)) + (eig_B.real**0.5).reshape(shape(1)))

t_pq = np.einsum('pq,pq->pq', D, omegas_pq)
E_2disp = -1*np.einsum('pq,pq', D, t_pq)

print('TDUHF DISP TOOK: %5.3f seconds\n' % ( time.time() - t))

# ref values taken from Hapka&Zuchowski molpro code
disp_ref = {
    'ArHF': {
            '3-21G' : -0.02331354,
            '6-31G' : -0.02793545,
            'aug-cc-pvdz' : -0.18362697,
    },
    'He2': {
            '3-21G' : -0.721440102e-4,
            '6-31G' : -0.12789154e-3,
            'aug-cc-pvdz' : -0.04751493,
    },
    'N2': {
            '3-21G' : -0.48765701e2,
            '6-31G' : -0.49075900e2,
            'aug-cc-pvdz' : -0.82041735e2,
    }
}

ind_ref = {
    'ArHF': {
            '3-21G' : -0.02331354,
            '6-31G' : -0.02793545,
            'aug-cc-pvdz' : -0.18362697
    },
    'He2': {
            '3-21G' : -0.721440102e-4,
            '6-31G' : -0.12789154e-3,
            'aug-cc-pvdz' : -0.04751493
    },
    'N2': {
            '3-21G' : -0.12918873e4,
            '6-31G' : -0.12607705e4,
            'aug-cc-pvdz' : -0.13127782e4,
    }
}

e_disp_ref = disp_ref.get(molecule, {}).get(basis, 1)

print("E^2_disp: {0} should be {1}".format(E_2disp*1000, e_disp_ref))
print("{}".format(E_2disp*1000/e_disp_ref))
