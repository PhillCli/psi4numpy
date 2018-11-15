# FIXME: Add proper citations
"""
Helper functions for the TDUHF response in SAPT directory.

References:
- Equations and algorithms from [Szalewicz:2005:43], [Jeziorski:1994:1887],
[Szalewicz:2012:254], and [Hohenstein:2012:304]
"""

__authors__   = "Daniel G. A. Smith"
__credits__   = ["Daniel G. A. Smith", "Filip Brzek"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2018-11-14"

import numpy as np
import time
import scipy.linalg as la

__all__ = ['tduhf_eigen']

def tduhf_eigen(occ, eps, v_eri, nov, coupled='True'):
    '''
    Implements generalized eigenvalue problem for TDUHF
    (A+B)(A-B)Z = w^{2}Z
    '''
    nocc, nvir = occ
    # alpha & beta
    nocc_a, nocc_b = nocc
    nvir_a, nvir_b = nvir

    eps_o, eps_v = eps
    # alpha & beta
    eps_o_a, eps_o_b = eps_o
    eps_v_a, eps_v_b = eps_v

    # different spin blocks
    v_ijab, v_iajb = v_eri
    v_ijab_aa, v_ijab_bb = v_ijab
    v_iajb_aa, v_iajb_bb, v_iajb_ab, v_iajb_ba = v_iajb

    nov_a, nov_b = nov

    t = time.time()
    # Building A and B blocks
    # 4 cases:
    # X_aa: (alpha, alpha),
    # X_ab: (alpha, beta),
    # X_ba: (beta,  alpha),
    # X_bb: (beta,  beta)


    # FIXME: DFT needs special treatment for v_xc kernel part
    #################
    # alpha & alpha #
    #################
    A_aa  = np.einsum('ab,ij->iajb', np.diag(eps_v_a), np.diag(np.ones(nocc_a)))
    A_aa -= np.einsum('ij,ab->iajb', np.diag(eps_o_a), np.diag(np.ones(nvir_a)))
    A_aa += v_iajb_aa
    A_aa -= v_ijab_aa.swapaxes(1, 2)
    print('[DEBUG] A_aa raw shape {}'.format(A_aa.shape))

    B_aa  = v_iajb_aa.copy()
    B_aa -= v_iajb_aa.swapaxes(1, 3)

    print('[DEBUG] B_aa raw shape {}'.format(B_aa.shape))

    A_aa.shape = (nov_a, nov_a)
    B_aa.shape = (nov_a, nov_a)
    print('[DEBUG] A/B_aa block shape {}'.format(A_aa.shape))

    ###############
    # beta & beta #
    ###############
    print("")
    A_bb  = np.einsum('ab,ij->iajb', np.diag(eps_v_b), np.diag(np.ones(nocc_b)))
    A_bb -= np.einsum('ij,ab->iajb', np.diag(eps_o_b), np.diag(np.ones(nvir_b)))
    A_bb += v_iajb_bb
    A_bb -= v_ijab_bb.swapaxes(1, 2)
    print('[DEBUG] A_bb raw shape {}'.format(A_bb.shape))

    B_bb  = v_iajb_bb.copy()
    B_bb -= v_iajb_bb.swapaxes(1, 3)

    print('[DEBUG] B_bb raw shape {}'.format(B_bb.shape))

    # Reshape and jam it together
    A_bb.shape = (nov_b, nov_b)
    B_bb.shape = (nov_b, nov_b)
    print('[DEBUG] A/B_bb block shape {}'.format(A_bb.shape))

    # No Exchange part for opposite spin electrons
    # B_(ai)(bj) = A_(ai)(bj) = (ai|bj)
    ################
    # alpha & beta #
    ################
    print("")
    A_ab = v_iajb_ab.copy()
    print('[DEBUG] A_ab raw shape {}'.format(A_ab.shape))

    B_ab  = v_iajb_ab.copy()
    print('[DEBUG] B_ab raw shape {}'.format(B_ab.shape))

    A_ab.shape = (nov_a, nov_b)
    B_ab.shape = (nov_a, nov_b)
    print('[DEBUG] A/B_ab block shape {}'.format(A_ab.shape))

    ################
    # beta & alpha #
    ################
    print("")
    A_ba = v_iajb_ba.copy()
    print('[DEBUG] A_ba raw shape {}'.format(A_ba.shape))

    B_ba  = v_iajb_ab.copy()
    print('[DEBUG] B_ba raw shape {}'.format(B_ba.shape))

    A_ba.shape = (nov_b, nov_a)
    B_ba.shape = (nov_b, nov_a)
    print('[DEBUG] A/B_ba block shape {}'.format(A_ba.shape))


    A1 = np.hstack((A_aa, A_ab))
    A2 = np.hstack((A_ba, A_bb))
    A = np.vstack((A1, A2))
    print('[DEBUG] A shape {}'.format(A.shape))

    B1 = np.hstack((B_aa, B_ab))
    B2 = np.hstack((B_ba, B_bb))
    B = np.vstack((B1, B2))
    print('[DEBUG] B shape {}'.format(B.shape))

    H1 = A + B
    H2 = A - B

    #print("H1: ")
    #print(H1)
    #print("H2: ")
    #print(H2)
    ##########################
    # TAKE CARE WE GOING HAM #
    # FOR He2 in 6-31G       #
    ##########################
    ##dimer = psi4.geometry("""
    ##He  0.000 0.000 -1.48169297
    ##--
    ##He  0.000 0.000  1.48169297
    ##""")

    #H1 = [[1.167223, -0.008676, -0.000107, 0.000340, 0.011439, 0.000070],
    #      [-0.008676, 1.684294, -0.004767, 0.011439, 0.455045, 0.003777],
    #      [-0.000107, -0.004767, 4.954139, 0.000070, 0.003777, 0.000046],
    #      [0.000340, 0.011439, 0.000070, 1.167233, -0.008676, -0.000107],
    #      [0.011439, 0.455045, 0.003777, -0.008676, 1.684294, -0.004767],
    #      [0.000070, 0.003777, 0.000046, -0.000107, -0.004767, 4.954139],]
    #H2 = [[1.167223, -0.008676, -0.000107, 0.0, 0.0, 0.0],
    #      [-0.008676, 1.684294, -0.004767, 0.0, 0.0, 0.0],
    #      [-0.000107, -0.004767, 4.954139, 0.0, 0.0, 0.0],
    #      [0.0, 0.0, 0.0, 1.167233, -0.008676, -0.000107],
    #      [0.0, 0.0, 0.0, -0.008676, 1.684294, -0.004767],
    #      [0.0, 0.0, 0.0, -0.000107, -0.004767, 4.954139],]

    # NOTE: This eigenproblem formulation is not Hermitian, thus
    # NOTE: less numerically stable.
    # TDUHF eigenproblem for omega and Z
    # (A-B)(A+B)Z = w**2 Z
    # Z = X + Y
    #LHS_Hess = np.dot(H2, H1)

    # NOTE: Hermitian Eigenproblem
    # Hirata, Gordon (1999)
    # 10.1016/S0009-2614(99)00137-2
    # (A-B)^(1/2)(A+B)(A-B)^(1/2) Z = w**2 Z
    # Z = (A-B)^(-1/2) (X+Y)
    H2_sqrt = la.sqrtm(H2)
    LHS_Hess = np.dot(H2_sqrt, H1)
    LHS_Hess = np.dot(LHS_Hess, H2_sqrt)

    # eigh matters
    eig, vect = la.eigh(LHS_Hess)

    # sorting doesn't change anything
    idx = eig.argsort()[::-1]
    eig = eig[idx]
    vect = vect[:, idx]

    print("[DEBUG]: Eigs shape: {}".format(eig.shape))
    print("[DEBUG]: Vect shape: {}".format(vect.shape))

    for v in range(nov_a + nov_b):
        w = eig[v].real**0.5
        w_sqrt = w**0.5
        z = vect[:,v].real
        # fixing Z normalization in hermitian eigenproblem
        z *= w**(-0.5)
        if abs(eig[v].imag) > 1e-13:
            print("[Warning] Large imaginary part! Im(omega) = {:3e}".format(eig[v].imag))

    print('Generalized eingenvalue took:  %5.3f seconds\n' % ( time.time() - t))

    # fixing Z normalization in hermitian eigenproblem
    vect = np.dot(H2_sqrt, vect)
    return eig, vect
