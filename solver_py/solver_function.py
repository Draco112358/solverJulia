import math as mt
import numpy as np
from scipy.sparse import csc_matrix, linalg
from numba import jit,njit, prange
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import lu_factor, lu_solve

class escals:
    def __init__(self,Lp,P,R,Cd,Is,Yle,freq):
        self.Lp = Lp
        self.P = P
        self.R = R
        self.Cd = Cd
        self.Is = Is
        self.Yle = Yle
        self.freq = freq

class out_class:
    def __init__(self, freq, S):
        self.freq = freq
        self.S = S

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

def ver_con(A,B):
    return np.vstack((A,B))

@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def compute_Z_self(R,Cd,w):
    len_R=R.shape[0]
    Z_self=np.zeros((len_R,1), dtype=np.complex_)

    for cont in prange(len_R):
        for aux in range(4):
            if R[cont,aux]!=0 and Cd[cont,aux]!=0:
                Z_self[cont]=Z_self[cont]+1.0/(1.0/R[cont,aux]+1j*w*Cd[cont,aux])
            else:
                if R[cont,aux]!=0:
                    Z_self[cont]=Z_self[cont]+R[cont,aux]
                else:
                    if Cd[cont,aux]!=0:
                        Z_self[cont]=Z_self[cont]+1.0/(1j*w*Cd[cont,aux])
    return Z_self

def build_Yle_S(lumped_elements, ports, escalings, n, w, val_chiusura):
    l1 = lumped_elements.le_nodes.shape[0]

    l2 = ports.port_nodes.shape[0]
    Ntot = 2 * l1 + 2 * l2
    contenitore = np.zeros((Ntot), dtype="int64")
    for k in range(l1):
        contenitore[k] = lumped_elements.le_nodes[k, 0]
        contenitore[k + l1] = lumped_elements.le_nodes[k, 1]
    for k in range(l2):
        contenitore[k + 2 * l1] = ports.port_nodes[k, 0]
        contenitore[k + 2 * l1 + l2] = ports.port_nodes[k, 1]

    N_ele = len(np.unique(contenitore))

    NNz_max = N_ele * N_ele

    ind_r = np.zeros((NNz_max), dtype='int64')
    ind_c = np.zeros((NNz_max), dtype='int64')
    vals = np.zeros((NNz_max), dtype=np.complex_)

    nlum = lumped_elements.le_nodes.shape[0]

    cont = 0
    for c1 in range(nlum):
        n1 = lumped_elements.le_nodes[c1, 0]
        n2 = lumped_elements.le_nodes[c1, 1]
        ind1 = np.where(ind_r == n1)
        ind2 = np.where(ind_c == n1)
        ind = np.intersect1d(ind1, ind2)
        if ind.size == 0:
            ind_r[cont] = n1
            ind_c[cont] = n1
            if lumped_elements.type[c1] == 2:
                vals[cont] = 1j * w * lumped_elements.value[c1]
            else:
                vals[cont] = 1.0 / lumped_elements.value[c1]
            cont = cont + 1
        else:
            if lumped_elements.type[c1] == 2:
                vals[ind[0]] = vals[ind[0]] + 1j * w * lumped_elements.value[c1]
            else:
                vals[ind[0]] = vals[ind[0]] + 1.0 / lumped_elements.value[c1]

        ind1 = np.where(ind_r == n2)
        ind2 = np.where(ind_c == n2)
        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            ind_r[cont] = n2
            ind_c[cont] = n2
            if lumped_elements.type[c1] == 2:
                vals[cont] = 1j * w * lumped_elements.value[c1]
            else:
                vals[cont] = 1.0 / lumped_elements.value[c1]
            cont = cont + 1
        else:
            if lumped_elements.type[c1] == 2:
                vals[ind[0]] = vals[ind[0]] + 1j * w * lumped_elements.value[c1]
            else:
                vals[ind[0]] = vals[ind[0]] + 1.0 / lumped_elements.value[c1]

        ind1 = np.where(ind_r == n1)
        ind2 = np.where(ind_c == n2)
        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            if lumped_elements.type[c1] == 2:
                vals[cont] = -1j * w * lumped_elements.value[c1]
            else:
                vals[cont] = -1.0 / lumped_elements.value[c1]
            ind_r[cont] = n1
            ind_c[cont] = n2
            cont = cont + 1

        else:
            if lumped_elements.type[c1] == 2:
                vals[ind[0]] = vals[ind[0]] - 1j * w * lumped_elements.value[c1]
            else:
                vals[ind[0]] = vals[ind[0]] - 1.0 / lumped_elements.value[c1]

        ind1 = np.where(ind_c == n1)
        ind2 = np.where(ind_r == n2)
        ind = np.intersect1d(ind1, ind2)
        if ind.size == 0:
            if lumped_elements.type[c1] == 2:
                vals[cont] = -1j * w * lumped_elements.value[c1]
            else:
                vals[cont] = -1.0 / lumped_elements.value[c1]
            ind_r[cont] = n2
            ind_c[cont] = n1
            cont = cont + 1
        else:
            if lumped_elements.type[c1] == 2:
                vals[ind[0]] = vals[ind[0]] - 1j * w * lumped_elements.value[c1]
            else:
                vals[ind[0]] = vals[ind[0]] - 1.0 / lumped_elements.value[c1]

    nport = ports.port_nodes.shape[0]

    for c1 in range(nport):

        n1 = ports.port_nodes[c1, 0]
        n2 = ports.port_nodes[c1, 1]

        ind1 = np.where(ind_r == n1)
        ind2 = np.where(ind_c == n1)

        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            ind_r[cont] = n1
            ind_c[cont] = n1
            vals[cont] = 1.0 / val_chiusura
            cont = cont + 1
        else:
            vals[ind[0]] = vals[ind[0]] + 1 / val_chiusura

        ind1 = np.where(ind_r == n2)
        ind2 = np.where(ind_c == n2)
        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            ind_r[cont] = n2
            ind_c[cont] = n2
            vals[cont] = 1.0 / val_chiusura
            cont = cont + 1
        else:
            vals[ind[0]] = vals[ind[0]] + 1.0 / val_chiusura

        ind1 = np.where(ind_r == n1)
        ind2 = np.where(ind_c == n2)
        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            vals[cont] = -1.0 / val_chiusura
            ind_r[cont] = n1
            ind_c[cont] = n2
            cont = cont + 1
        else:
            vals[ind[0]] = vals[ind[0]] - 1.0 / val_chiusura

        ind1 = np.where(ind_r == n2)
        ind2 = np.where(ind_c == n1)
        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            vals[cont] = -1.0 / val_chiusura
            ind_r[cont] = n2
            ind_c[cont] = n1
            cont = cont + 1
        else:
            vals[ind[0]] = vals[ind[0]] - 1.0 / val_chiusura

    Yle = csc_matrix((escalings.Yle*vals[0:cont], (ind_r[0:cont], ind_c[0:cont])), shape=(n, n))

    return Yle

def build_Yle_S_no_scal(lumped_elements, ports, n, w, val_chiusura):

    l1 = lumped_elements.le_nodes.shape[0]
            
    
    l2 = ports.port_nodes.shape[0]
    Ntot = 2 * l1 + 2 * l2
    contenitore = np.zeros((Ntot), dtype="int64")
    for k in range(l1):
        contenitore[k] = lumped_elements.le_nodes[k, 0]
        contenitore[k + l1] = lumped_elements.le_nodes[k, 1]
    for k in range(l2):
        contenitore[k + 2 * l1] = ports.port_nodes[k, 0]
        contenitore[k + 2 * l1 + l2] = ports.port_nodes[k, 1]

    N_ele = len(np.unique(contenitore))

    NNz_max = N_ele * N_ele

    ind_r = np.zeros((NNz_max), dtype='int64')
    ind_c = np.zeros((NNz_max), dtype='int64')
    vals = np.zeros((NNz_max), dtype=np.complex_)

    nlum = lumped_elements.le_nodes.shape[0]

    cont = 0
    for c1 in range(nlum):
        n1 = lumped_elements.le_nodes[c1, 0]
        n2 = lumped_elements.le_nodes[c1, 1]
        ind1 = np.where(ind_r == n1)
        ind2 = np.where(ind_c == n1)
        ind = np.intersect1d(ind1, ind2)
        if ind.size == 0:
            ind_r[cont] = n1
            ind_c[cont] = n1
            if lumped_elements.type[c1] == 2:
                vals[cont] = 1j * w * lumped_elements.value[c1]
            else:
                vals[cont] = 1.0 / lumped_elements.value[c1]
            cont = cont + 1
        else:
            if lumped_elements.type[c1] == 2:
                vals[ind[0]] = vals[ind[0]] + 1j * w * lumped_elements.value[c1]
            else:
                vals[ind[0]] = vals[ind[0]] + 1.0 / lumped_elements.value[c1]

        ind1 = np.where(ind_r == n2)
        ind2 = np.where(ind_c == n2)
        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            ind_r[cont] = n2
            ind_c[cont] = n2
            if lumped_elements.type[c1] == 2:
                vals[cont] = 1j * w * lumped_elements.value[c1]
            else:
                vals[cont] = 1.0 / lumped_elements.value[c1]
            cont = cont + 1
        else:
            if lumped_elements.type[c1] == 2:
                vals[ind[0]] = vals[ind[0]] + 1j * w * lumped_elements.value[c1]
            else:
                vals[ind[0]] = vals[ind[0]] + 1.0 / lumped_elements.value[c1]

        ind1 = np.where(ind_r == n1)
        ind2 = np.where(ind_c == n2)
        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            if lumped_elements.type[c1] == 2:
                vals[cont] = -1j * w * lumped_elements.value[c1]
            else:
                vals[cont] = -1.0 / lumped_elements.value[c1]
            ind_r[cont] = n1
            ind_c[cont] = n2
            cont = cont + 1

        else:
            if lumped_elements.type[c1] == 2:
                vals[ind[0]] = vals[ind[0]] - 1j * w * lumped_elements.value[c1]
            else:
                vals[ind[0]] = vals[ind[0]] - 1.0 / lumped_elements.value[c1]

        ind1 = np.where(ind_c == n1)
        ind2 = np.where(ind_r == n2)
        ind = np.intersect1d(ind1, ind2)
        if ind.size == 0:
            if lumped_elements.type[c1] == 2:
                vals[cont] = -1j * w * lumped_elements.value[c1]
            else:
                vals[cont] = -1.0 / lumped_elements.value[c1]
            ind_r[cont] = n2
            ind_c[cont] = n1
            cont = cont + 1
        else:
            if lumped_elements.type[c1] == 2:
                vals[ind[0]] = vals[ind[0]] - 1j * w * lumped_elements.value[c1]
            else:
                vals[ind[0]] = vals[ind[0]] - 1.0 / lumped_elements.value[c1]

    nport = ports.port_nodes.shape[0]

    for c1 in range(nport):

        n1 = ports.port_nodes[c1, 0]
        n2 = ports.port_nodes[c1, 1]

        ind1 = np.where(ind_r == n1)
        ind2 = np.where(ind_c == n1)

        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            ind_r[cont] = n1
            ind_c[cont] = n1
            vals[cont] = 1.0 / val_chiusura
            cont = cont + 1
        else:
            vals[ind[0]] = vals[ind[0]] + 1 / val_chiusura

        ind1 = np.where(ind_r == n2)
        ind2 = np.where(ind_c == n2)
        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            ind_r[cont] = n2
            ind_c[cont] = n2
            vals[cont] = 1.0 / val_chiusura
            cont = cont + 1
        else:
            vals[ind[0]] = vals[ind[0]] + 1.0 / val_chiusura

        ind1 = np.where(ind_r == n1)
        ind2 = np.where(ind_c == n2)
        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            vals[cont] = -1.0 / val_chiusura
            ind_r[cont] = n1
            ind_c[cont] = n2
            cont = cont + 1
        else:
            vals[ind[0]] = vals[ind[0]] - 1.0 / val_chiusura

        ind1 = np.where(ind_r == n2)
        ind2 = np.where(ind_c == n1)
        ind = np.intersect1d(ind1, ind2)

        if ind.size == 0:
            vals[cont] = -1.0 / val_chiusura
            ind_r[cont] = n2
            ind_c[cont] = n1
            cont = cont + 1
        else:
            vals[ind[0]] = vals[ind[0]] - 1.0 / val_chiusura

    Yle = csc_matrix((vals[0:cont], (ind_r[0:cont], ind_c[0:cont])), shape=(n, n))

    return Yle

def precond_3_3_vector(LU_S,invZ,invP,A,Gamma,w,X1,X2,X3):

    n1=len(X1)
    n2=len(X2)
    n3=len(X3)

    i1=range(n1)
    i2=range(n1,n1+n2)
    i3=range(n1+n2,n1+n2+n3)

    Y=np.zeros((n1+n2+n3,1), dtype=np.complex_)

    M1 = csc_matrix.dot(invZ, X1)
    M2 = LU_S.solve(csc_matrix.dot(A.transpose(),M1))
    M3 = csc_matrix.dot(invP,X2)
    M4 = LU_S.solve(csc_matrix.dot(Gamma,M3))
    M5 = LU_S.solve(X3)

    Y[np.ix_(i1)] = Y[np.ix_(i1)]+M1-1.0*csc_matrix.dot(invZ,csc_matrix.dot(A,M2))
    Y[np.ix_(i1)] = Y[np.ix_(i1)]+1j*w*csc_matrix.dot(invZ, csc_matrix.dot(A,M4))
    Y[np.ix_(i1)] = Y[np.ix_(i1)]-1.0*csc_matrix.dot(invZ, csc_matrix.dot(A, M5))

    Y[np.ix_(i2)] = Y[np.ix_(i2)]+csc_matrix.dot(invP, csc_matrix.dot(Gamma.transpose(), M2))
    Y[np.ix_(i2)] = Y[np.ix_(i2)] + M3 -1j*w*csc_matrix.dot(invP, csc_matrix.dot(Gamma.transpose(), M4))
    Y[np.ix_(i2)] = Y[np.ix_(i2)]+csc_matrix.dot(invP, csc_matrix.dot(Gamma.transpose(), M5))

    Y[np.ix_(i3)] = Y[np.ix_(i3)]+M2
    Y[np.ix_(i3)] = Y[np.ix_(i3)]-1j*w*M4
    Y[np.ix_(i3)] = Y[np.ix_(i3)]+M5

    return Y

def precond_3_3_Kt(LU_S, invZ, invP, A,Gamma, n1,n2, X3):

    n3 = len(X3)

    i1 = range(n1)
    i2 = range(n1, n1 + n2)
    i3 = range(n1 + n2, n1 + n2 + n3)

    Y = np.zeros((n1 + n2 + n3,1), dtype=np.complex_)

    M5 = LU_S.solve(X3)

    Y[np.ix_(i1)] = Y[np.ix_(i1)] - 1.0*csc_matrix.dot(invZ, csc_matrix.dot(A, M5))

    Y[np.ix_(i2)] = Y[np.ix_(i2)] + csc_matrix.dot(invP, csc_matrix.dot(Gamma.transpose(), M5))

    Y[np.ix_(i3)] = Y[np.ix_(i3)] + M5

    return Y

import time

def s2z(S,Zo):
    num_ports=S.shape[0]
    nfreq=S.shape[2]
    Z = np.zeros((num_ports, num_ports, nfreq), dtype=np.complex_)
    Id = np.identity(num_ports)
    for cont in range(nfreq):
        Z[:,:,cont]=Zo*np.linalg.solve(Id-1.0*S[:,:,cont], Id+S[:,:,cont])
    return Z

def s2y(S,Zo):
    num_ports=S.shape[0]
    nfreq=S.shape[2]
    Y = np.zeros((num_ports, num_ports, nfreq), dtype=np.complex_)
    Id = np.identity(num_ports)
    for cont in range(nfreq):
        Y[:,:,cont]=Zo*np.linalg.solve(Id+S[:,:,cont], Id-1.0*S[:,:,cont])
    return Y


def ComputeMatrixVector(w,escalings,A,Gamma,P_mat,Lp_x_mat,Lp_y_mat,Lp_z_mat,
                        Z_self,Yle,invZ,invP,LU_S,x):

    mx = Lp_x_mat.shape[0]
    my = Lp_y_mat.shape[0]
    mz = Lp_z_mat.shape[0]

    m = mx + my + mz
    ns = Gamma.shape[1]
    n = Gamma.shape[0]
    I = np.zeros((m, 1), dtype=np.complex_)
    Q = np.zeros((ns, 1), dtype=np.complex_)
    Phi = np.zeros((n, 1), dtype=np.complex_)
    I[0:m, 0] = x[0:m]
    Q[0:ns, 0] = x[m:m + ns]
    Phi[0:n, 0] = x[m + ns:m + ns + n]
    Y1 = np.zeros((m, 1), dtype=np.complex_)

    ia1 = range(mx)
    ia2 = range(mx, mx + my)
    ia3 = range(mx + my, mx + my + mz)

    Y1[ia1] = 1j * w * escalings.Lp * np.dot(Lp_x_mat, I[ia1])
    Y1[ia2] = 1j * w * escalings.Lp * np.dot(Lp_y_mat, I[ia2])
    Y1[ia3] = 1j * w * escalings.Lp * np.dot(Lp_z_mat, I[ia3])

    Y1=Y1+np.multiply(Z_self,I)+csc_matrix.dot(A, Phi)

    Y2 = escalings.P * np.dot(P_mat,Q) -1.0*csc_matrix.dot(Gamma.transpose(), Phi)
    Y3 = -1.0*(csc_matrix.dot(A.transpose(), I))+ \
         csc_matrix.dot(Yle, Phi) +1j*w*(csc_matrix.dot(Gamma, Q))

    MatrixVector = precond_3_3_vector(LU_S, invZ, invP, A,Gamma, w, Y1, Y2, Y3)

    return MatrixVector

def Quasi_static_iterative_solver(freq_in,A,Gamma,P_mat,Lp_x_mat,\
    Lp_y_mat,Lp_z_mat,diag_R,diag_Cd,ports,lumped_elements,GMRES_settings):

    escalings = escals(1e6, 1e-12, 1e-3, 1e12, 1e3, 1e3, 1e-9)
    #escalings = escals(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    freq = freq_in * escalings.freq
    # GMRES settings - ---------------------------
    Inner_Iter = GMRES_settings.Inner_Iter
    Outer_Iter = GMRES_settings.Outer_Iter
    # -------------------------------------------

    mx = Lp_x_mat.shape[0]
    my = Lp_y_mat.shape[0]
    mz = Lp_z_mat.shape[0]

    m = mx + my + mz
    n = A.shape[1]
    ns = Gamma.shape[1]

    w = 2 * mt.pi * freq

    nfreq = w.shape[0]

    Is = np.zeros((n,1), dtype='double')

    num_ports=ports.port_start.shape[0]

    S = np.zeros((num_ports, num_ports, nfreq), dtype=np.complex_)

    X_prec=np.zeros((m+n+ns, num_ports), dtype=np.complex_)

    diag_P = np.zeros((ns), dtype='double')
    for c in range(ns):
        diag_P[c]=escalings.P*P_mat[c,c]

    diag_Lp = np.zeros((m), dtype='double')
    for c in range(mx):
        diag_Lp[c] = escalings.Lp*Lp_x_mat[c, c]
    for c in range(my):
        diag_Lp[c+mx] = escalings.Lp*Lp_y_mat[c, c]
    for c in range(mz):
        diag_Lp[c+mx+my] = escalings.Lp*Lp_z_mat[c, c]

    invP =csc_matrix((1. / diag_P, (range(ns), range(ns))), shape=(ns, ns))

    val_chiusura = 50.0

    diag_R=escalings.R*diag_R
    diag_Cd=escalings.Cd * diag_Cd

    for k in range(nfreq):
        print('Freq n=', k+1, ' - Freq Tot=', nfreq)
        Z_self = compute_Z_self(diag_R,diag_Cd, w[k])
        Yle = build_Yle_S(lumped_elements, ports, escalings, n, w[k] / escalings.freq, val_chiusura)
        invZ = csc_matrix((1. / (Z_self[:,0] + 1j * w[k] * diag_Lp), (range(m), range(m))), shape=(m, m))

        SS=Yle+csc_matrix.dot(A.transpose(),\
          csc_matrix.dot(invZ,A))+1j*w[k]*csc_matrix.dot(Gamma,\
          csc_matrix.dot(invP,Gamma.transpose()))

        LU_S=linalg.splu(SS,options=dict(Equil=False, IterRefine='SINGLE',SymmetricMode=True))
        #LU_S = linalg.spilu(SS, drop_tol=1e-6, options=dict(SymmetricMode=True))

        for c1 in range(num_ports):
            n1 = ports.port_nodes[c1, 0]
            n2 = ports.port_nodes[c1, 1]
            Is[n1] = 1.0 * escalings.Is
            Is[n2] = -1.0 * escalings.Is

            tn = precond_3_3_Kt(LU_S, invZ, invP, A,Gamma, m, ns, Is)

            counter = gmres_counter()

            products_law = lambda x: \
                ComputeMatrixVector( w[k], escalings, A,Gamma, P_mat,Lp_x_mat,Lp_y_mat,Lp_z_mat,
                                    Z_self, Yle, invZ,invP, LU_S,x)

            prodts = LinearOperator((n + m + ns, n + m + ns), products_law)

            V_o, info = linalg.gmres(prodts, tn, x0=X_prec[:, c1], tol=GMRES_settings.tol[k], \
                                          restart=None, maxiter=Inner_Iter, M=None, \
                                          callback=counter, restrt=None, atol=None, callback_type=None)

            V = np.array(V_o)
            X_prec[:, c1]=V

            Is[n1] = 0.0
            Is[n2] = 0.0

            if info == 0:
                print("convergence reached, number of iterations: ", counter.niter)
            else:
                if info > 0:
                    print("convergence not reached, number of iterations: ", counter.niter)
                else:
                    print("illegal input or breakdown, number of iterations: ", counter.niter)

            for c2 in range(num_ports):
                n3 = ports.port_nodes[c2, 0]
                n4 = ports.port_nodes[c2, 1]

                if c1 == c2:
                    S[c1, c2, k] = (2.0 * (V[m + ns + n3] - V[m + ns + n4]) - val_chiusura) / val_chiusura
                else:
                    S[c1, c2, k] = (2.0 * (V[m + ns + n3] - V[m + ns + n4])) / val_chiusura

                S[c2, c1, k] = S[c1, c2, k]

    Z=s2z(S,val_chiusura)
    Y=s2y(S,val_chiusura)
    return Z,Y,S

def hor_con(A,B):
    return np.hstack((A,B))

def ver_con(A,B):
    return np.vstack((A,B))

def Quasi_static_direct_solver(freq,A,Gamma,P_mat,Lp_x_mat,\
    Lp_y_mat,Lp_z_mat,diag_R,diag_Cd,ports,lumped_elements):

    mx = Lp_x_mat.shape[0]
    my = Lp_y_mat.shape[0]
    mz = Lp_z_mat.shape[0]

    m = mx + my + mz
    n = A.shape[1]
    ns = Gamma.shape[1]

    w = 2 * mt.pi * freq

    nfreq = w.shape[0]

    Is = np.zeros((n,1), dtype='double')

    num_ports=ports.port_start.shape[0]

    S = np.zeros((num_ports, num_ports, nfreq), dtype=np.complex_)

    val_chiusura = 50.0

    Lp = ver_con(ver_con( \
            hor_con(hor_con(Lp_x_mat, np.zeros((mx, my), dtype='double')), np.zeros((mx, mz), dtype='double')), \
            hor_con(hor_con(np.zeros((my, mx), dtype='double'), Lp_y_mat), np.zeros((my, mz), dtype='double')) \
            ), \
            hor_con(hor_con(np.zeros((mz, mx), dtype='double'), np.zeros((mz, my), dtype='double')), Lp_z_mat) \
            )

    A = csc_matrix.todense(A)
    Gamma = csc_matrix.todense(Gamma)

    for k in range(nfreq):
        print('Freq n =', k+1, ' - Freq Tot =', nfreq)
        Z_self = compute_Z_self(diag_R,diag_Cd, w[k])
        Yle = build_Yle_S_no_scal(lumped_elements, ports, n, w[k], val_chiusura)
        Yle = csc_matrix.todense(Yle)

        Matrix = ver_con(ver_con( \
            hor_con(hor_con(np.diag(Z_self.flatten()) + 1j * w[k] * Lp, np.zeros((m, ns), dtype='double')), A), \
            hor_con(hor_con(np.zeros((ns, m), dtype='double'), P_mat),-1.0 * Gamma.transpose())), \
                hor_con(hor_con(-1.0 * A.transpose(), 1j * (w[k] * Gamma)), Yle) \
            )

        lu, piv = lu_factor(Matrix)

        for c1 in range(num_ports):
            n1 = ports.port_nodes[c1, 0]
            n2 = ports.port_nodes[c1, 1]
            Is[n1] = 1.0
            Is[n2] = -1.0

            tn = ver_con(ver_con( \
                np.zeros((m, 1), dtype='double'), \
                np.zeros((ns, 1), dtype='double') \
                ), \
                Is \
                )

            V = lu_solve((lu, piv), tn)

            Is[n1] = 0.0
            Is[n2] = 0.0


            for c2 in range(num_ports):
                n3 = ports.port_nodes[c2, 0]
                n4 = ports.port_nodes[c2, 1]

                if c1 == c2:
                    S[c1, c2, k] = (2.0 * (V[m + ns + n3] - V[m + ns + n4]) - val_chiusura) / val_chiusura
                else:
                    S[c1, c2, k] = (2.0 * (V[m + ns + n3] - V[m + ns + n4])) / val_chiusura

                S[c2, c1, k] = S[c1, c2, k]

    print(S.shape)
    Z = s2z(S, val_chiusura)
    Y = s2y(S, val_chiusura)
    return Z, Y, S


