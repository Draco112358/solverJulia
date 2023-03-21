import numpy as np
# import scipy.io
import numba as nmb
import scipy.sparse as scisparse # csc_matrix,csr_matrix



@nmb.jit(nopython=True, cache=True, fastmath=True)
def From_3D_to_1D(i, j, k, M, N):
    pos = ((k) * M * N) + ((j) * M) + i
    return pos

@nmb.jit(nopython=True, cache=True, fastmath=True)
def bin_search(num, A):

    index = 0
    n = A.shape[0]
    left = 0
    right = n-1

    while left <= right:
        mid = np.int64(np.ceil((left + right) / 2))

        if A[mid] == num:
            index = mid
            break
        else:
            if A[mid] > num:
                right = mid - 1
            else:
                left = mid + 1

    return index

@nmb.jit(nopython=True, cache=True, fastmath=True)
def create_volumes_mapping_and_centers(matrice,Nx,Ny,Nz,num_centri,sx,sy,sz,min_v):

    print("----",(Nx * Ny * Nz))
 
    mapping = np.zeros((Nx * Ny * Nz), dtype='int64')
    centri_vox = np.zeros((num_centri, 3), dtype='double')
    id_mat = np.zeros((num_centri), dtype='int64')

    num_grids = matrice.shape[0]
    num_ele=0
    for cont in range(Nx):
        for cont2 in range(Ny):
            for cont3 in range(Nz):
                for k in range(num_grids):
                    if(matrice[k,cont,cont2,cont3]==1):
                        mapping[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]=num_ele
                        centri_vox[num_ele][0] = min_v[0] + sx * (cont)  + sx / 2.0
                        centri_vox[num_ele][1] = min_v[1] + sy * (cont2) + sy / 2.0
                        centri_vox[num_ele][2] = min_v[2] + sz * (cont3) + sz / 2.0
                        id_mat[num_ele] = k
                        num_ele=num_ele+1
                        break

    return num_ele,mapping,centri_vox,id_mat

@nmb.jit(nopython=True, cache=True, fastmath=True)
def create_nodes_ref(matrice, Nx,Ny,Nz, num_centri, external_g, m_volumes):

    num_grids = matrice.shape[0]
    nodes = np.zeros((num_centri), dtype='int64')

    for cont in range(Nx):
        for cont2 in range(Ny):
            for cont3 in range(Nz):
                for k in range(num_grids):
                    if (matrice[k, cont, cont2, cont3] == 1):
                        c1 = 2 * cont + 1
                        c2 = 2 * cont2 + 1
                        c3 = 2 * cont3 + 1
                        f1 = external_g[k, 0, cont, cont2, cont3]
                        f2 = external_g[k, 1, cont, cont2, cont3]
                        f3 = external_g[k, 2, cont, cont2, cont3]
                        f4 = external_g[k, 3, cont, cont2, cont3]
                        f5 = external_g[k, 4, cont, cont2, cont3]
                        f6 = external_g[k, 5, cont, cont2, cont3]
                        is_f1 = f1
                        is_f2 = f2
                        if (f1==1 and f2==1):
                            is_f1 = 0
                            is_f2 = 0

                        is_f3 = f3
                        is_f4 = f4
                        if (f3==1 and f4==1):
                            is_f3 = 0
                            is_f4 = 0

                        is_f5 = f5
                        is_f6 = f6
                        if (f5 == 1 and f6 == 1):
                            is_f5 = 0
                            is_f6 = 0

                        if (is_f1==1 or is_f2==1 or is_f3==1 or is_f4==1 or is_f5==1 or is_f6==1) :
                            if (is_f1==1 and is_f2==0 and is_f3==0 and is_f4==0 and is_f5==0 and is_f6==0) :
                                nodes[m_volumes[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]] = \
                                    From_3D_to_1D(c1, c2 - 1, c3, 3 * Nx, 3 * Ny)
                            else:
                                if (is_f2 == 1 and is_f1 == 0 and is_f3 == 0 and is_f4 == 0 and is_f5 == 0 and is_f6 == 0):
                                    nodes[m_volumes[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]] = \
                                        From_3D_to_1D(c1, c2 + 1, c3, 3 * Nx, 3 * Ny)
                                else:
                                    if (is_f3 == 1 and is_f1 == 0 and is_f2 == 0 and is_f4 == 0 and is_f5 == 0 and is_f6 == 0):
                                        nodes[m_volumes[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]] = \
                                            From_3D_to_1D(c1-1, c2 , c3, 3 * Nx, 3 * Ny)
                                    else:
                                        if (is_f4 == 1 and is_f1 == 0 and is_f2 == 0 and is_f3 == 0 and is_f5 == 0 and is_f6 == 0):
                                            nodes[m_volumes[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]] = \
                                                From_3D_to_1D(c1 + 1, c2, c3, 3 * Nx, 3 * Ny)
                                        else:
                                            if (is_f5 == 1 and is_f1 == 0 and is_f2 == 0 and is_f3 == 0 and is_f4 == 0 and is_f6 == 0):
                                                nodes[m_volumes[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]] = \
                                                    From_3D_to_1D(c1, c2, c3-1, 3 * Nx, 3 * Ny)
                                            else:
                                                if (is_f6 == 1 and is_f1 == 0 and is_f2 == 0 and is_f3 == 0 and is_f4 == 0 and is_f5 == 0):
                                                    nodes[m_volumes[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]] = \
                                                        From_3D_to_1D(c1, c2, c3 + 1, 3 * Nx, 3 * Ny)
                                                else:
                                                    if (is_f1 == 1 and is_f3 == 1 and is_f2 == 0 and is_f4 == 0 and is_f5 == 0 and is_f6 == 0):
                                                        nodes[m_volumes[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]] = \
                                                            From_3D_to_1D(c1-1, c2-1, c3, 3 * Nx, 3 * Ny)
                                                    else:
                                                        if (is_f1 == 1 and is_f4 == 1 and is_f2 == 0 and is_f3 == 0 and is_f5 == 0 and is_f6 == 0):
                                                            nodes[m_volumes[
                                                                From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]] = \
                                                                From_3D_to_1D(c1 + 1, c2 - 1, c3, 3 * Nx, 3 * Ny)
                                                        else:
                                                            if (is_f1 == 1 and is_f5 == 1 and is_f2 == 0 and is_f3 == 0 and is_f4 == 0 and is_f6 == 0):
                                                                nodes[m_volumes[
                                                                    From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]] = \
                                                                    From_3D_to_1D(c1, c2 - 1, c3 - 1, 3 * Nx, 3 * Ny)
                                                            else:
                                                                if (is_f1 == 1 and is_f6 == 1 and is_f2 == 0 and is_f3 == 0 and is_f4 == 0 and is_f5 == 0):
                                                                    nodes[m_volumes[
                                                                        From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]] = \
                                                                        From_3D_to_1D(c1, c2 - 1, c3 + 1, 3 * Nx,3 * Ny)
                                                                else:
                                                                    if (is_f1 == 1 and is_f3 == 1 and is_f5 == 1 and is_f2 == 0 and is_f4 == 0 and is_f6 == 0):
                                                                        nodes[m_volumes[From_3D_to_1D(cont, cont2, cont3, Nx,Ny)]] = \
                                                                            From_3D_to_1D(c1-1, c2 - 1, c3 - 1, 3 * Nx, 3 * Ny)
                                                                    else:
                                                                        if (is_f1 == 1 and is_f3 == 1 and is_f6 == 1 and is_f2 == 0 and is_f4 == 0 and is_f5 == 0):
                                                                            nodes[m_volumes[From_3D_to_1D(cont, cont2, cont3, Nx,Ny)]] = \
                                                                                From_3D_to_1D(c1 - 1, c2 - 1, c3 + 1,3 * Nx, 3 * Ny)
                                                                        else:
                                                                            if (is_f1 == 1 and is_f4 == 1 and is_f5 == 1 and is_f2 == 0 and is_f3 == 0 and is_f6 == 0):
                                                                                nodes[m_volumes[From_3D_to_1D(cont, cont2, cont3,Nx, Ny)]] = \
                                                                                    From_3D_to_1D(c1 + 1, c2 - 1,c3 - 1, 3 * Nx,3 * Ny)
                                                                            else:
                                                                                if (is_f1 == 1 and is_f4 == 1 and is_f6 == 1 and is_f2 == 0 and is_f3 == 0 and is_f5 == 0):
                                                                                    nodes[m_volumes[From_3D_to_1D(cont, cont2,cont3, Nx, Ny)]] = \
                                                                                        From_3D_to_1D(c1 + 1, c2 - 1,c3 + 1, 3 * Nx,3 * Ny)
                                                                                else:
                                                                                    if (is_f2 == 1 and is_f3 == 1 and is_f1 == 0 and is_f4 == 0 and is_f5 == 0 and is_f6 == 0):
                                                                                        nodes[m_volumes[From_3D_to_1D(cont, cont2,cont3, Nx,Ny)]] = \
                                                                                            From_3D_to_1D(c1 - 1,c2 + 1,c3,3 * Nx,3 * Ny)
                                                                                    else:
                                                                                        if (is_f2 == 1 and is_f4 == 1 and is_f1 == 0 and is_f3 == 0 and is_f5 == 0 and is_f6 == 0):
                                                                                            nodes[m_volumes[From_3D_to_1D(cont,cont2,cont3, Nx,Ny)]] = \
                                                                                                From_3D_to_1D(c1 + 1,c2 + 1,c3,3 * Nx,3 * Ny)
                                                                                        else:
                                                                                            if (is_f2 == 1 and is_f5 == 1 and is_f1 == 0 and is_f3 == 0 and is_f4 == 0 and is_f6 == 0):
                                                                                                nodes[m_volumes[From_3D_to_1D(cont,cont2,cont3,Nx,Ny)]] = \
                                                                                                    From_3D_to_1D(c1, c2 + 1,c3 - 1, 3 * Nx,3 * Ny)
                                                                                            else:
                                                                                                if (is_f2 == 1 and is_f6 == 1 and is_f1 == 0 and is_f3 == 0 and is_f4 == 0 and is_f5 == 0):
                                                                                                    nodes[m_volumes[From_3D_to_1D(cont, cont2,cont3, Nx,Ny)]] = \
                                                                                                        From_3D_to_1D(c1, c2 + 1,c3 + 1,3 * Nx,3 * Ny)
                                                                                                else:
                                                                                                    if (is_f2 == 1 and is_f3 == 1 and is_f5 == 1 and is_f1 == 0 and is_f4 == 0 and is_f6 == 0):
                                                                                                        nodes[m_volumes[From_3D_to_1D(cont,cont2,cont3,Nx,Ny)]] = \
                                                                                                            From_3D_to_1D(c1 - 1,c2 + 1,c3 - 1,3 * Nx,3 * Ny)
                                                                                                    else:
                                                                                                        if (is_f2 == 1 and is_f3 == 1 and is_f6 == 1 and is_f1 == 0 and is_f4 == 0 and is_f5 == 0):
                                                                                                            nodes[m_volumes[From_3D_to_1D(cont,cont2,cont3,Nx,Ny)]] = \
                                                                                                                From_3D_to_1D(c1 - 1,c2 + 1,c3 + 1, 3 * Nx, 3 * Ny)
                                                                                                        else:
                                                                                                            if (is_f2 == 1 and is_f4 == 1 and is_f5 == 1 and is_f1 == 0 and is_f3 == 0 and is_f6 == 0):
                                                                                                                nodes[m_volumes[From_3D_to_1D(cont,cont2,cont3,Nx,Ny)]] = \
                                                                                                                    From_3D_to_1D(c1 + 1,c2 + 1,c3 - 1,3 * Nx,3 * Ny)
                                                                                                            else:
                                                                                                                if (is_f2 == 1 and is_f4 == 1 and is_f6 == 1 and is_f1 == 0 and is_f3 == 0 and is_f5 == 0):
                                                                                                                    nodes[m_volumes[From_3D_to_1D( cont,cont2,cont3,Nx, Ny)]] = \
                                                                                                                        From_3D_to_1D(c1 + 1,c2 + 1,c3 + 1,3 * Nx, 3 * Ny)
                                                                                                                else:
                                                                                                                    if (is_f3 == 1 and is_f5 == 1 and is_f1 == 0 and is_f2 == 0 and is_f4 == 0 and is_f6 == 0):
                                                                                                                        nodes[m_volumes[From_3D_to_1D(cont,cont2,cont3,Nx,Ny)]] = \
                                                                                                                            From_3D_to_1D(c1 - 1,c2 ,c3 - 1,3 * Nx,3 * Ny)
                                                                                                                    else:
                                                                                                                        if (is_f3 == 1 and is_f6 == 1 and is_f1 == 0 and is_f2 == 0 and is_f4 == 0 and is_f5 == 0):
                                                                                                                            nodes[m_volumes[From_3D_to_1D(cont,cont2,cont3,Nx,Ny)]] = \
                                                                                                                                From_3D_to_1D(c1 - 1,c2, c3 + 1,3 * Nx,3 * Ny)
                                                                                                                        else:
                                                                                                                            if (is_f4 == 1 and is_f5 == 1 and is_f1 == 0 and is_f2 == 0 and is_f3 == 0 and is_f6 == 0):
                                                                                                                                nodes[m_volumes[From_3D_to_1D(cont,cont2,cont3,Nx,Ny)]] = \
                                                                                                                                    From_3D_to_1D(c1 + 1,c2,c3 - 1,3 * Nx,3 * Ny)
                                                                                                                            else:
                                                                                                                                if (is_f4 == 1 and is_f6 == 1 and is_f1 == 0 and is_f2 == 0 and is_f3 == 0 and is_f5 == 0):
                                                                                                                                    nodes[m_volumes[ From_3D_to_1D(cont,cont2,cont3,Nx,Ny)]] = \
                                                                                                                                        From_3D_to_1D(c1 + 1,c2,c3 + 1,3 * Nx,3 * Ny)
                        else:
                            nodes[m_volumes[From_3D_to_1D( cont,cont2,cont3,Nx, Ny)]] = From_3D_to_1D(c1,c2,c3,3 * Nx, 3 * Ny)

                        break
    nodes_red = np.unique(nodes)
    return nodes_red,nodes

@nmb.jit(nopython=True, cache=True, fastmath=True)
def distfcm(center, data):

    out=np.power(np.sum((np.power((data-np.ones((data.shape[0],1))*center),2)),axis=1),0.5)
    return out

@nmb.jit(nopython=True, cache=True, fastmath=True)
def nodes_find_rev(Nodes_inp_coord, nodi_centri, node_to_skip):

    indici = np.argsort(distfcm(Nodes_inp_coord, nodi_centri))

    if (indici[0]!=node_to_skip):
        node = indici[0]
    else:
        node = indici[1]

    return node

@nmb.jit(nopython=True, cache=True, fastmath=True)
def find_nodes_port(nodi_centri, port_start, port_end, nodi, nodi_red):

    N = port_start.shape[0]
    port_voxels = np.zeros((N, 2), dtype='int64')
    port_nodes = np.zeros((N, 2), dtype='int64')

    for cont in nmb.prange(N):
        port_voxels[cont, 0] = nodes_find_rev(port_start[cont,:], nodi_centri, -1)
        port_voxels[cont, 1] = nodes_find_rev(port_end[cont,:],nodi_centri, port_voxels[cont,0])
        port_nodes[cont, 0] = bin_search(nodi[port_voxels[cont, 0]],nodi_red)
        port_nodes[cont, 1] = bin_search(nodi[port_voxels[cont, 1]],nodi_red)

    return port_voxels,port_nodes

@nmb.jit(nopython=True, cache=True, fastmath=True)
def create_external_grids(matrice,Nx,Ny,Nz):
    num_grids = matrice.shape[0]
    
    OUTPUTgrids = np.zeros((num_grids,6,Nx,Ny,Nz), dtype='int8')

    for cont2 in range(Ny):
        for cont3 in range(Nz):
            for k in range(num_grids):
                if(matrice[k][0][cont2][cont3]==1):
                    OUTPUTgrids[k,2,0,cont2,cont3]=1

    for cont2 in range(Ny):
        for cont3 in range(Nz):
            for k in range(num_grids):
                if(matrice[k][Nx-1][cont2][cont3]==1):
                    OUTPUTgrids[k,3,(Nx-1),cont2,cont3]=1

    for cont in range(Nx):
        for cont3 in range(Nz):
            for k in range(num_grids):
                if(matrice[k][cont][0][cont3]==1):
                    OUTPUTgrids[k,0,cont,0,cont3] = 1

    for cont in range(Nx):
        for cont3 in range(Nz):
            for k in range(num_grids):
                if (matrice[k][cont][Ny-1][cont3] == 1):
                    OUTPUTgrids[k,1,cont,Ny-1,cont3] = 1

    for cont in range(Nx):
        for cont2 in range(Ny):
            for k in range(num_grids):
                if(matrice[k][cont][cont2][0]==1):
                    OUTPUTgrids[k,4,cont,cont2,0] = 1

    for cont in range(Nx):
        for cont2 in range(Ny):
            for k in range(num_grids):
                if(matrice[k][cont][cont2][Nz-1]==1):
                    OUTPUTgrids[k,5,cont,cont2,Nz-1] = 1

    for cont in range(1,Nx-1):
        for cont2 in range(Ny):
            for cont3 in range(Nz):
                for k in range(num_grids):
                    if (matrice[k][cont][cont2][cont3] == 1):
                        if (matrice[k][cont - 1][cont2][cont3] == 0):
                            OUTPUTgrids[k,2,cont,cont2,cont3] = 1
                        if (matrice[k][cont + 1][cont2][cont3] == 0):
                            OUTPUTgrids[k,3,cont,cont2,cont3] = 1

    for cont in range(Nx):
        for cont2 in range(1,Ny-1):
            for cont3 in range(Nz):
                for k in range(num_grids):
                    if (matrice[k][cont][cont2][cont3] == 1):
                        if (matrice[k][cont][cont2 -1][cont3] == 0):
                            OUTPUTgrids[k,0,cont,cont2,cont3] = 1
                        if (matrice[k][cont][cont2 + 1][cont3] == 0):
                            OUTPUTgrids[k,1,cont,cont2,cont3] = 1

    for cont in range(Nx):
        for cont2 in range(Ny):
            for cont3 in range(1,Nz-1):
                for k in range(num_grids):
                    if (matrice[k][cont][cont2][cont3] == 1):
                        if (matrice[k][cont][cont2][cont3 - 1] == 0):
                            OUTPUTgrids[k,4,cont,cont2,cont3] = 1

                        if (matrice[k][cont][cont2][cont3 + 1] == 0):
                            OUTPUTgrids[k,5,cont,cont2,cont3] = 1

    return OUTPUTgrids

@nmb.jit(nopython=True, cache=True, fastmath=True)
def create_mapping_Ax(matrice,Nx,Ny,Nz):
    num_grids = matrice.shape[0]

    N_max = ((Nx - 1) * Ny * Nz)
    mapping = np.zeros((N_max), dtype='int64')

    num_ele = 0

    for cont2 in range(Ny):
        for cont3 in range(Nz):
            for cont in range(Nx-1):
                for k in range(num_grids):
                    if(matrice[k,cont,cont2,cont3]==1) and (matrice[k,cont+1,cont2,cont3]==1):
                        kkey = From_3D_to_1D(cont, cont2, cont3, Nx - 1, Ny)
                        if mapping[kkey] == 0:
                            mapping[kkey] = num_ele
                            num_ele = num_ele + 1
                        break

    return mapping,num_ele

@nmb.jit(nopython=True, cache=True, fastmath=True)
def create_mapping_Ay(matrice,Nx,Ny,Nz):
    num_grids = matrice.shape[0]

    N_max = ( Nx * (Ny - 1) * Nz)
    mapping = np.zeros((N_max), dtype='int64')

    num_ele = 0

    for cont3 in range(Nz):
        for cont in range(Nx):
            for cont2 in range(Ny - 1):
                for k in range(num_grids):
                    if(matrice[k,cont,cont2,cont3]==1) and (matrice[k,cont,cont2+1,cont3]==1):
                        kkey = From_3D_to_1D(cont, cont2, cont3, Nx, Ny - 1)
                        if mapping[kkey] == 0:
                           mapping[kkey] = num_ele
                           num_ele = num_ele + 1
                        break

    return mapping,num_ele

@nmb.jit(nopython=True, cache=True, fastmath=True)
def create_mapping_Az(matrice,Nx,Ny,Nz):
    num_grids = matrice.shape[0]
    N_max = ( Nx * Ny * (Nz - 1) )
    mapping = np.zeros((N_max), dtype='int64')

    num_ele = 0

    for cont in range(Nx):
        for cont2 in range(Ny):
            for cont3 in range(Nz - 1):
                for k in range(num_grids):
                    if(matrice[k,cont,cont2,cont3]==1) and (matrice[k,cont,cont2,cont3+1]==1):
                        kkey = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        if mapping[kkey] == 0:
                           mapping[kkey] = num_ele
                           num_ele = num_ele + 1
                        break

    return mapping,num_ele

@nmb.jit(nopython=True, cache=True, fastmath=True)
def create_A_mats_volInd(matrice,Nx,Ny,Nz,mapping_Vox,\
                    mapAx, NAx, mapAy, NAy, mapAz, NAz, sx,sy,sz,min_v,nodi,nodi_red):
 
    num_grids = matrice.shape[0]
    lix_mat = np.zeros((NAx, 2), dtype='int8')
    lix_border = np.zeros((NAx, 2), dtype='int8')
    ind_row = np.zeros((2*NAx+2*NAy+2*NAz), dtype='int64')
    ind_col = np.zeros((2*NAx+2*NAy+2*NAz), dtype='int64')
    vals_A = np.zeros((2*NAx+2*NAy+2*NAz), dtype='double')
    bars_Lp_x = np.zeros((NAx, 6), dtype='double')

    num_ele = 0

    for cont2 in range(Ny):
        for cont3 in range(Nz):
            for cont in range(Nx - 1):
                for k in range(num_grids):
                    if (matrice[k, cont, cont2, cont3] == 1) and (matrice[k, cont+1, cont2, cont3] == 1):

                        pos = mapAx[From_3D_to_1D(cont, cont2, cont3, Nx - 1, Ny)]

                        ind_row[num_ele] = pos
                        ind_col[num_ele] = bin_search(nodi[mapping_Vox[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]], nodi_red)

                        bars_Lp_x[pos][0] = min_v[0] + sx * cont + sx/2
                        bars_Lp_x[pos][1] = min_v[1] + sy * cont2
                        bars_Lp_x[pos][2] = min_v[2] + sz * cont3

                        vals_A[num_ele] = -1.0
                        num_ele = num_ele + 1

                        ind_row[num_ele] = pos
                        ind_col[num_ele] = bin_search(nodi[mapping_Vox[From_3D_to_1D(cont+1, cont2, cont3, Nx, Ny)]], nodi_red)
                        vals_A[num_ele] = 1.0
                        num_ele = num_ele + 1

                        bars_Lp_x[pos][3] = bars_Lp_x[pos][0] + sx
                        bars_Lp_x[pos][4] = bars_Lp_x[pos][1] + sy
                        bars_Lp_x[pos][5] = bars_Lp_x[pos][2] + sz

                        lix_mat[pos, 0] = k+1
                        lix_mat[pos, 1] = k+1

                        if cont > 0:
                           if (matrice[k,cont-1, cont2, cont3]==0):
                               lix_border[pos, 0] = k + 1
                               bars_Lp_x[pos][0] = bars_Lp_x[pos][0] - sx/2.0
                        else:
                            lix_border[pos, 0] = k + 1
                            bars_Lp_x[pos][0] = bars_Lp_x[pos][0] - sx / 2.0

                        if cont + 1 == Nx-1:
                            lix_border[pos, 1] = k + 1
                            bars_Lp_x[pos][3] = bars_Lp_x[pos][3] + sx/2.0
                        else:
                            if cont + 2 <= Nx-1:
                                if (matrice[k,cont+2, cont2, cont3]==0):
                                    lix_border[pos, 1] = k + 1
                                    bars_Lp_x[pos][3] = bars_Lp_x[pos][3] + sx/2.0
                        break


    liy_mat = np.zeros((NAy, 2), dtype='int8')
    liy_border = np.zeros((NAy, 2), dtype='int8')
    bars_Lp_y = np.zeros((NAy, 6), dtype='double')

    starter=num_ele
    num_ele = 0

    for cont3 in range(Nz):
        for cont in range(Nx):
            for cont2 in range(Ny - 1):
                for k in range(num_grids):
                    if (matrice[k, cont, cont2, cont3] == 1) and (matrice[k, cont, cont2+1, cont3] == 1):

                        pos = mapAy[From_3D_to_1D(cont, cont2, cont3, Nx, Ny - 1)]

                        bars_Lp_y[pos][0] = min_v[0] + sx * cont
                        bars_Lp_y[pos][1] = min_v[1] + sy * cont2 + sy/2
                        bars_Lp_y[pos][2] = min_v[2] + sz * cont3

                        ind_row[starter+num_ele] = pos+NAx
                        ind_col[starter+num_ele] = bin_search(nodi[mapping_Vox[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]], nodi_red)
                        vals_A[starter+num_ele] = -1.0
                        num_ele = num_ele + 1

                        bars_Lp_y[pos][3] = bars_Lp_y[pos][0] + sx
                        bars_Lp_y[pos][4] = bars_Lp_y[pos][1] + sy
                        bars_Lp_y[pos][5] = bars_Lp_y[pos][2] + sz

                        ind_row[starter+num_ele] = pos+NAx
                        ind_col[starter+num_ele] = bin_search(nodi[mapping_Vox[From_3D_to_1D(cont, cont2+1, cont3, Nx, Ny)]], nodi_red)
                        vals_A[starter+num_ele] = 1.0
                        num_ele = num_ele + 1

                        liy_mat[pos, 0] = k+1
                        liy_mat[pos, 1] = k+1

                        if cont2 > 0:
                           if (matrice[k,cont, cont2-1, cont3]==0):
                               liy_border[pos, 0] = k + 1
                               bars_Lp_y[pos][1] = bars_Lp_y[pos][1] - sy/2.0
                        else:
                            liy_border[pos, 0] = k + 1
                            bars_Lp_y[pos][1] = bars_Lp_y[pos][1] - sy / 2.0

                        if cont2 + 1 == Ny-1:
                            liy_border[pos, 1] = k + 1
                            bars_Lp_y[pos][4] = bars_Lp_y[pos][4] + sy / 2.0
                        else:
                            if cont2 + 2 <= Ny-1:
                                if (matrice[k,cont, cont2+2, cont3]==0):
                                    liy_border[pos, 1] = k + 1
                                    bars_Lp_y[pos][4] = bars_Lp_y[pos][4] + sy / 2.0
                        break


    liz_mat = np.zeros((NAz, 2), dtype='int8')
    liz_border = np.zeros((NAz, 2), dtype='int8')
    bars_Lp_z = np.zeros((NAz, 6), dtype='double')

    starter = starter+num_ele
    num_ele = 0

    for cont in range(Nx):
        for cont2 in range(Ny):
            for cont3 in range(Nz - 1):
                for k in range(num_grids):
                    if (matrice[k, cont, cont2, cont3] == 1) and (matrice[k, cont, cont2, cont3+1] == 1):

                        pos = mapAz[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]

                        bars_Lp_z[pos][0] = min_v[0] + sx * cont
                        bars_Lp_z[pos][1] = min_v[1] + sy * cont2
                        bars_Lp_z[pos][2] = min_v[2] + sz * cont3 + sz/2

                        ind_row[starter+num_ele] = pos+NAx+NAy
                        ind_col[starter+num_ele] = bin_search(nodi[mapping_Vox[From_3D_to_1D(cont, cont2, cont3, Nx, Ny)]], nodi_red)
                        vals_A[starter+num_ele] = -1.0
                        num_ele = num_ele + 1

                        bars_Lp_z[pos][3] = bars_Lp_z[pos][0] + sx
                        bars_Lp_z[pos][4] = bars_Lp_z[pos][1] + sy
                        bars_Lp_z[pos][5] = bars_Lp_z[pos][2] + sz

                        ind_row[starter+num_ele] = pos+NAx+NAy
                        ind_col[starter+num_ele] = bin_search(nodi[mapping_Vox[From_3D_to_1D(cont, cont2, cont3+1, Nx, Ny)]], nodi_red)
                        vals_A[starter+num_ele] = 1.0
                        num_ele = num_ele + 1

                        liz_mat[pos, 0] = k+1
                        liz_mat[pos, 1] = k+1

                        if cont3 > 0:
                           if (matrice[k,cont, cont2, cont3-1]==0):
                               liz_border[pos, 0] = k + 1
                               bars_Lp_z[pos][2] = bars_Lp_z[pos][2] - sz/2.0
                        else:
                            liz_border[pos, 0] = k + 1
                            bars_Lp_z[pos][2] = bars_Lp_z[pos][2] - sz / 2.0

                        if cont3 + 1 == Nz-1:
                            liz_border[pos, 1] = k + 1
                            bars_Lp_z[pos][5] = bars_Lp_z[pos][5] + sz / 2.0
                        else:
                            if cont3 + 2 <= Nz-1:
                                if (matrice[k,cont, cont2, cont3+2]==0):
                                    liz_border[pos, 1] = k + 1
                                    bars_Lp_z[pos][5] = bars_Lp_z[pos][5] + sz / 2.0
                        break


    return ind_row,ind_col,vals_A,lix_mat,liy_mat,liz_mat,lix_border,liy_border,liz_border,bars_Lp_x,bars_Lp_y,bars_Lp_z

def ver_con(A,B):
    return np.vstack((A,B))

def vect_con(A,B):
    return np.hstack((A,B))

def compute_diagonals(MATER,sx,sy,sz,lix_mat,liy_mat,liz_mat,lix_border,liy_border,liz_border):
    eps0 = 8.854187816997944e-12
    num_grids = len(MATER)
    for cont in range(num_grids):
        sigmar = MATER[cont].conductivity
        epsr = MATER[cont].permittivity 
        if sigmar!=0:
            MATER[cont].Rx = 0.5 * sx / (sigmar * sy * sz)
            MATER[cont].Ry = 0.5 * sy / (sigmar * sx * sz)
            MATER[cont].Rz = 0.5 * sz / (sigmar * sy * sx)
            if epsr == 1:
                MATER[cont].Cx = 0.0
                MATER[cont].Cy = 0.0
                MATER[cont].Cz = 0.0
            else:
                MATER[cont].Cx = eps0 * (epsr - 1.0) * sy * sz / (0.5 * sx)
                MATER[cont].Cy = eps0 * (epsr - 1.0) * sx * sz / (0.5 * sy)
                MATER[cont].Cz = eps0 * (epsr - 1.0) * sy * sx / (0.5 * sz)
        else:
            MATER[cont].Rx = 0.0
            MATER[cont].Ry = 0.0
            MATER[cont].Rz = 0.0
            MATER[cont].Cx = eps0 * (epsr - 1.0) * sy * sz / (0.5 * sx)
            MATER[cont].Cy = eps0 * (epsr - 1.0) * sx * sz / (0.5 * sy)
            MATER[cont].Cz = eps0 * (epsr - 1.0) * sy * sx / (0.5 * sz)

    Rx = np.zeros((lix_border.shape[0], 4), dtype='complex')
    Ry = np.zeros((liy_border.shape[0], 4), dtype='complex')
    Rz = np.zeros((liz_border.shape[0], 4), dtype='complex')

    Cx = np.zeros((lix_border.shape[0], 4), dtype='complex')
    Cy = np.zeros((liy_border.shape[0], 4), dtype='complex')
    Cz = np.zeros((liz_border.shape[0], 4), dtype='complex')

    for cont in range(num_grids):
        if MATER[cont].Rx!=0:
            ind_m = np.where((cont+1) == lix_mat[:, 0])
            Rx[ind_m, 0] = MATER[cont].Rx
            ind_m = np.where((cont+1) == lix_mat[:, 1])
            Rx[ind_m, 1] = MATER[cont].Rx
            ind_m = np.where((cont+1) == lix_border[:, 0])
            Rx[ind_m, 2] = MATER[cont].Rx
            ind_m = np.where((cont+1) == lix_border[:, 1])
            Rx[ind_m, 3] = MATER[cont].Rx
        if MATER[cont].Cx!=0:
            ind_m = np.where((cont+1) == lix_mat[:, 0])
            Cx[ind_m, 0] = MATER[cont].Cx
            ind_m = np.where((cont+1) == lix_mat[:, 1])
            Cx[ind_m, 1] = MATER[cont].Cx
            ind_m = np.where((cont+1) == lix_border[:, 0])
            Cx[ind_m, 2] = MATER[cont].Cx
            ind_m = np.where((cont+1) == lix_border[:, 1])
            Cx[ind_m, 3] = MATER[cont].Cx

        if MATER[cont].Ry!=0:
            ind_m = np.where((cont+1) == liy_mat[:, 0])
            Ry[ind_m, 0] = MATER[cont].Ry
            ind_m = np.where((cont+1) == liy_mat[:, 1])
            Ry[ind_m, 1] = MATER[cont].Ry
            ind_m = np.where((cont+1) == liy_border[:, 0])
            Ry[ind_m, 2] = MATER[cont].Ry
            ind_m = np.where((cont+1) == liy_border[:, 1])
            Ry[ind_m, 3] = MATER[cont].Ry
        if MATER[cont].Cy!=0:
            ind_m = np.where((cont+1) == liy_mat[:, 0])
            Cy[ind_m, 0] = MATER[cont].Cy
            ind_m = np.where((cont+1) == liy_mat[:, 1])
            Cy[ind_m, 1] = MATER[cont].Cy
            ind_m = np.where((cont+1) == liy_border[:, 0])
            Cy[ind_m, 2] = MATER[cont].Cy
            ind_m = np.where((cont+1) == liy_border[:, 1])
            Cy[ind_m, 3] = MATER[cont].Cy

        if MATER[cont].Rz!=0:
            ind_m = np.where((cont+1) == liz_mat[:, 0])
            Rz[ind_m, 0] = MATER[cont].Rz
            ind_m = np.where((cont+1) == liz_mat[:, 1])
            Rz[ind_m, 1] = MATER[cont].Rz
            ind_m = np.where((cont+1) == liz_border[:, 0])
            Rz[ind_m, 2] = MATER[cont].Rz
            ind_m = np.where((cont+1) == liz_border[:, 1])
            Rz[ind_m, 3] = MATER[cont].Rz
        if MATER[cont].Cz!=0:
            ind_m = np.where((cont+1) == liz_mat[:, 0])
            Cz[ind_m, 0] = MATER[cont].Cz
            ind_m = np.where((cont+1) == liz_mat[:, 1])
            Cz[ind_m, 1] = MATER[cont].Cz
            ind_m = np.where((cont+1) == liz_border[:, 0])
            Cz[ind_m, 2] = MATER[cont].Cz
            ind_m = np.where((cont+1) == liz_border[:, 1])
            Cz[ind_m, 3] = MATER[cont].Cz

    diag_R = ver_con(ver_con(Rx, Ry), Rz)
    diag_Cd = ver_con(ver_con(Cx, Cy), Cz)

    return diag_R, diag_Cd

@nmb.jit(nopython=True, cache=True, fastmath=True)
def create_Gamma_and_center_sup(matrice, Nx,Ny,Nz, map_volumes, min_v, \
                                sx, sy, sz, nodi, nodi_red, ext_grids):
    num_grids = matrice.shape[0]
    mapping_surf_1 = np.zeros((Nx * Ny * Nz), dtype='int64')
    mapping_surf_2 = np.zeros((Nx * Ny * Nz), dtype='int64')
    mapping_surf_3 = np.zeros((Nx * Ny * Nz), dtype='int64')
    mapping_surf_4 = np.zeros((Nx * Ny * Nz), dtype='int64')
    mapping_surf_5 = np.zeros((Nx * Ny * Nz), dtype='int64')
    mapping_surf_6 = np.zeros((Nx * Ny * Nz), dtype='int64')

    num_ele_1 = 0
    num_ele_2 = 0
    num_ele_3 = 0
    num_ele_4 = 0
    num_ele_5 = 0
    num_ele_6 = 0

    nnz_surf_max = 6 * Nx * Ny * Nz
    ind_r = np.zeros((nnz_surf_max), dtype='int64')
    ind_c = np.zeros((nnz_surf_max), dtype='int64')

    sup_centers = np.zeros((nnz_surf_max,3), dtype='double')
    sup_type = np.zeros((nnz_surf_max), dtype='int64')

    contat_tot = 0

    cont2=0
    for cont in range(Nx):
        for cont3 in range(Nz):
            for k in range(num_grids):
                if (matrice[k,cont, cont2, cont3]==1) and (ext_grids[k, 1,cont, cont2, cont3]==0):
                    num_ele_1 = num_ele_1 + 1
                    p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                    mapping_surf_1[p31] = num_ele_1-1
                    contat_tot = contat_tot + 1
                    ind_r[contat_tot-1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                    ind_c[contat_tot-1] = num_ele_1-1
                    sup_centers[ind_c[contat_tot-1], 0] = min_v[0] + sx * cont  + sx / 2.0
                    sup_centers[ind_c[contat_tot-1], 1] = min_v[1] + sy * cont2
                    sup_centers[ind_c[contat_tot-1], 2] = min_v[2] + sz * cont3 + sz / 2.0
                    sup_type[ind_c[contat_tot-1]] = 1
                    break

    for cont in range(Nx):
        for cont2 in range(1,Ny):
            for cont3 in range(Nz):
                for k in range(num_grids):
                    if (matrice[k,cont, cont2, cont3]==1) and (matrice[k,cont, cont2-1, cont3]==0) and (ext_grids[k, 1,cont, cont2, cont3]==0):
                        num_ele_1 = num_ele_1 + 1
                        p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        mapping_surf_1[p31] = num_ele_1 - 1
                        contat_tot = contat_tot + 1
                        ind_r[contat_tot-1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                        ind_c[contat_tot-1] = num_ele_1 - 1
                        sup_centers[ind_c[contat_tot-1], 0] = min_v[0] + sx * cont + sx / 2.0
                        sup_centers[ind_c[contat_tot-1], 1] = min_v[1] + sy * cont2
                        sup_centers[ind_c[contat_tot-1], 2] = min_v[2] + sz * cont3 + sz / 2.0
                        sup_type[ind_c[contat_tot-1]] = 1
                        break

    cont2 = Ny - 1
    for cont in range(Nx):
        for cont3 in range(Nz):
            for k in range(num_grids):
                if (matrice[k, cont, cont2, cont3] == 1) and (ext_grids[k, 0, cont, cont2, cont3] == 0):
                    num_ele_2 = num_ele_2 + 1
                    p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                    mapping_surf_2[p31] = num_ele_2 - 1
                    contat_tot = contat_tot + 1
                    ind_r[contat_tot-1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                    ind_c[contat_tot-1] = num_ele_1 + num_ele_2 - 1
                    sup_centers[ind_c[contat_tot-1], 0] = min_v[0] + sx * cont + sx / 2.0
                    sup_centers[ind_c[contat_tot-1], 1] = min_v[1] + sy * cont2 + sy
                    sup_centers[ind_c[contat_tot-1], 2] = min_v[2] + sz * cont3 + sz / 2.0
                    sup_type[ind_c[contat_tot-1]] = 1
                    break

    for cont in range(Nx):
        for cont2 in range(0,Ny-1):
            for cont3 in range(Nz):
                for k in range(num_grids):
                    if (matrice[k, cont, cont2, cont3] == 1) and (matrice[k, cont, cont2+1, cont3] == 0) and (ext_grids[k, 0, cont, cont2, cont3] == 0):
                        check_others = False
                        for k2 in range(num_grids):
                            if k!=k2:
                                if (matrice[k2,cont, cont2+1, cont3]==1):
                                    check_others = True
                                    break

                        if not check_others:
                            num_ele_2 = num_ele_2 + 1
                            p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                            mapping_surf_2[p31] = num_ele_2 - 1
                            contat_tot = contat_tot + 1
                            ind_r[contat_tot-1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                            ind_c[contat_tot-1] = num_ele_1 + num_ele_2 - 1
                            sup_centers[ind_c[contat_tot-1], 0] = min_v[0] + sx * cont + sx / 2.0
                            sup_centers[ind_c[contat_tot-1], 1] = min_v[1] + sy * cont2 + sy
                            sup_centers[ind_c[contat_tot-1], 2] = min_v[2] + sz * cont3 + sz / 2.0
                            sup_type[ind_c[contat_tot-1]] = 1

                        break


    cont = 0
    for cont2 in range(Ny):
        for cont3 in range(Nz):
            for k in range(num_grids):
                if (matrice[k, cont, cont2, cont3] == 1) and (ext_grids[k, 3, cont, cont2, cont3] == 0):
                    num_ele_3 = num_ele_3 + 1
                    p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                    mapping_surf_3[p31] = num_ele_3 - 1
                    contat_tot = contat_tot + 1
                    ind_r[contat_tot - 1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                    ind_c[contat_tot - 1] = num_ele_1 + num_ele_2 + num_ele_3 - 1
                    sup_centers[ind_c[contat_tot - 1], 0] = min_v[0] + sx * cont
                    sup_centers[ind_c[contat_tot - 1], 1] = min_v[1] + sy * cont2 + sy / 2.0
                    sup_centers[ind_c[contat_tot - 1], 2] = min_v[2] + sz * cont3 + sz / 2.0
                    sup_type[ind_c[contat_tot - 1]] = 2
                    break

    for cont in range(1,Nx):
        for cont2 in range(Ny):
            for cont3 in range(Nz):
                for k in range(num_grids):
                    if (matrice[k,cont, cont2, cont3]==1) and (matrice[k,cont-1, cont2, cont3]==0) and (ext_grids[k, 3,cont, cont2, cont3]==0):
                        num_ele_3 = num_ele_3 + 1
                        p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        mapping_surf_3[p31] = num_ele_3 - 1
                        contat_tot = contat_tot + 1
                        ind_r[contat_tot-1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                        ind_c[contat_tot-1] = num_ele_1 + num_ele_2 + num_ele_3 - 1
                        sup_centers[ind_c[contat_tot-1], 0] = min_v[0] + sx * cont
                        sup_centers[ind_c[contat_tot-1], 1] = min_v[1] + sy * cont2 + sy / 2.0
                        sup_centers[ind_c[contat_tot-1], 2] = min_v[2] + sz * cont3 + sz / 2.0
                        sup_type[ind_c[contat_tot-1]] = 2
                        break

    cont = Nx - 1
    for cont2 in range(Ny):
        for cont3 in range(Nz):
            for k in range(num_grids):
                if (matrice[k, cont, cont2, cont3] == 1) and (ext_grids[k, 2, cont, cont2, cont3] == 0):
                    num_ele_4 = num_ele_4 + 1
                    p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                    mapping_surf_4[p31] = num_ele_4 - 1
                    contat_tot = contat_tot + 1
                    ind_r[contat_tot-1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                    ind_c[contat_tot-1] = num_ele_1 + num_ele_2 + num_ele_3 + num_ele_4 - 1
                    sup_centers[ind_c[contat_tot-1], 0] = min_v[0] + sx * cont + sx
                    sup_centers[ind_c[contat_tot-1], 1] = min_v[1] + sy * cont2 + sy / 2.0
                    sup_centers[ind_c[contat_tot-1], 2] = min_v[2] + sz * cont3 + sz / 2.0
                    sup_type[ind_c[contat_tot-1]] = 2
                    break

    for cont in range(0,Nx-1):
        for cont2 in range(0,Ny):
            for cont3 in range(Nz):
                for k in range(num_grids):
                    if (matrice[k, cont, cont2, cont3] == 1) and (matrice[k, cont+1, cont2, cont3] == 0) and (ext_grids[k, 2, cont, cont2, cont3] == 0):
                        check_others = False
                        for k2 in range(num_grids):
                            if k!=k2:
                                if (matrice[k2,cont+1, cont2, cont3]==1):
                                    check_others = True
                                    break

                        if not check_others:
                            num_ele_4 = num_ele_4 + 1
                            p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                            mapping_surf_4[p31] = num_ele_4 - 1
                            contat_tot = contat_tot + 1
                            ind_r[contat_tot-1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                            ind_c[contat_tot-1] = num_ele_1 + num_ele_2 + num_ele_3 + num_ele_4 - 1
                            sup_centers[ind_c[contat_tot-1], 0] = min_v[0] + sx * cont + sx
                            sup_centers[ind_c[contat_tot-1], 1] = min_v[1] + sy * cont2 + sy / 2.0
                            sup_centers[ind_c[contat_tot-1], 2] = min_v[2] + sz * cont3 + sz / 2.0
                            sup_type[ind_c[contat_tot-1]] = 2

                        break



    cont3 = 0
    for cont in range(Nx):
        for cont2 in range(Ny):
            for k in range(num_grids):
                if (matrice[k, cont, cont2, cont3] == 1) and (ext_grids[k, 5, cont, cont2, cont3] == 0):
                    num_ele_5 = num_ele_5 + 1
                    p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                    mapping_surf_5[p31] = num_ele_5 - 1
                    contat_tot = contat_tot + 1
                    ind_r[contat_tot - 1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                    ind_c[contat_tot - 1] = num_ele_1 + num_ele_2 + num_ele_3 + num_ele_4 + num_ele_5 - 1
                    sup_centers[ind_c[contat_tot - 1], 0] = min_v[0] + sx * cont + sx / 2.0
                    sup_centers[ind_c[contat_tot - 1], 1] = min_v[1] + sy * cont2 + sy / 2.0
                    sup_centers[ind_c[contat_tot - 1], 2] = min_v[2] + sz * cont3
                    sup_type[ind_c[contat_tot - 1]] = 3
                    break

    for cont in range(Nx):
        for cont2 in range(Ny):
            for cont3 in range(1,Nz):
                for k in range(num_grids):
                    if (matrice[k,cont, cont2, cont3]==1) and (matrice[k,cont, cont2, cont3-1]==0) and (ext_grids[k, 5,cont, cont2, cont3]==0):
                        num_ele_5 = num_ele_5 + 1
                        p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                        mapping_surf_5[p31] = num_ele_5 - 1
                        contat_tot = contat_tot + 1
                        ind_r[contat_tot-1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                        ind_c[contat_tot-1] = num_ele_1 + num_ele_2 + num_ele_3 + num_ele_4 + num_ele_5 - 1
                        sup_centers[ind_c[contat_tot-1], 0] = min_v[0] + sx * cont + sx / 2.0
                        sup_centers[ind_c[contat_tot-1], 1] = min_v[1] + sy * cont2 + sy / 2.0
                        sup_centers[ind_c[contat_tot-1], 2] = min_v[2] + sz * cont3
                        sup_type[ind_c[contat_tot-1]] = 3
                        break

    cont3 = Nz - 1
    for cont in range(Nx):
        for cont2 in range(Ny):
            for k in range(num_grids):
                if (matrice[k, cont, cont2, cont3] == 1) and (ext_grids[k, 4, cont, cont2, cont3] == 0):
                    num_ele_6 = num_ele_6 + 1
                    p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                    mapping_surf_6[p31] = num_ele_6 - 1
                    contat_tot = contat_tot + 1
                    ind_r[contat_tot - 1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                    ind_c[contat_tot - 1] = num_ele_1 + num_ele_2 + num_ele_3 + num_ele_4 + num_ele_5 + num_ele_6 - 1
                    sup_centers[ind_c[contat_tot - 1], 0] = min_v[0] + sx * cont + sx / 2.0
                    sup_centers[ind_c[contat_tot - 1], 1] = min_v[1] + sy * cont2 + sy / 2.0
                    sup_centers[ind_c[contat_tot - 1], 2] = min_v[2] + sz * cont3 + sz
                    sup_type[ind_c[contat_tot - 1]] = 3
                    break

    for cont in range(Nx):
        for cont2 in range(Ny):
            for cont3 in range(0,Nz-1):
                for k in range(num_grids):
                    if (matrice[k, cont, cont2, cont3] == 1) and (matrice[k, cont, cont2, cont3+1] == 0) and (ext_grids[k, 4, cont, cont2, cont3] == 0):
                        check_others = False
                        for k2 in range(num_grids):
                            if k!=k2:
                                if (matrice[k2,cont, cont2, cont3+1]==1):
                                    check_others = True
                                    break

                        if not check_others:
                            num_ele_6 = num_ele_6 + 1
                            p31 = From_3D_to_1D(cont, cont2, cont3, Nx, Ny)
                            mapping_surf_6[p31] = num_ele_6 - 1
                            contat_tot = contat_tot + 1
                            ind_r[contat_tot-1] = bin_search(nodi[map_volumes[p31]], nodi_red)
                            ind_c[contat_tot-1] = num_ele_1 + num_ele_2 + num_ele_3 + num_ele_4 + num_ele_5 + num_ele_6 - 1
                            sup_centers[ind_c[contat_tot-1], 0] = min_v[0] + sx * cont + sx / 2.0
                            sup_centers[ind_c[contat_tot-1], 1] = min_v[1] + sy * cont2 + sy / 2.0
                            sup_centers[ind_c[contat_tot-1], 2] = min_v[2] + sz * cont3 + sz
                            sup_type[ind_c[contat_tot-1]] = 3

                        break

    nnz_surf = num_ele_1 + num_ele_2 + num_ele_3 + num_ele_4 + num_ele_5 + num_ele_6
    vals = np.ones((contat_tot), dtype='double')
    ind_r = ind_r[0:contat_tot]
    ind_c = ind_c[0:contat_tot]
    sup_centers = sup_centers[0:nnz_surf,:]
    sup_type = sup_type[0:nnz_surf]

    return vals,ind_r,ind_c,sup_centers,sup_type

def generate_interconnection_matrices_and_centers(size_x,size_y,size_z,\
                                                  grid_matrix,num_cel_x,num_cel_y,num_cel_z,\
                                                  materials,port_matrix,lumped_el_matrix,minimum_vertex):

    numTotVox = num_cel_x*num_cel_y*num_cel_z
    
    print('Total Number of Voxels (including air):', numTotVox)
    n_grids = len(materials)
    assert grid_matrix.shape[0]==n_grids
    num_tot_full_vox = 0
    
    for i in range(n_grids):
        num_tot_full_vox = num_tot_full_vox + np.count_nonzero(grid_matrix[i])
    # TODO: VERIFICARE SE OCCORRE
    cont=0
    while cont<len(materials):
        materials[cont].epsr = materials[cont].permittivity +1j * materials[cont].permittivity * materials[cont].tangent_delta_permittivity
        cont+=1

    assert type(num_cel_x)==int
    assert type(num_cel_y)==int
    assert type(num_cel_z)==int

    n_boxes,mapping_vols,volume_centers,volumes_materials=\
        create_volumes_mapping_and_centers(matrice=grid_matrix,Nx=num_cel_x,Ny=num_cel_y,Nz=num_cel_z,\
                                            num_centri=num_tot_full_vox,sx=size_x,sy=size_y,sz=size_z,min_v=minimum_vertex)

    externals_grids = create_external_grids(matrice=grid_matrix,Nx=num_cel_x,Ny=num_cel_y,Nz=num_cel_z)
    
    nodes_red, nodes = create_nodes_ref(matrice=grid_matrix,Nx=num_cel_x,Ny=num_cel_y,Nz=num_cel_z,\
                                        num_centri=num_tot_full_vox, external_g=externals_grids, m_volumes=mapping_vols)

    port_matrix.voxels, port_matrix.nodes = find_nodes_port(nodi_centri=volume_centers, port_start=port_matrix.port_start, port_end=port_matrix.port_end, nodi=nodes, nodi_red=nodes_red)
        


    lumped_el_matrix.voxels, lumped_el_matrix.nodes = find_nodes_port(nodi_centri=volume_centers,\
                                                                      port_start=lumped_el_matrix.le_start,
                                                                      port_end=lumped_el_matrix.le_end, nodi=nodes, nodi_red=nodes_red)


    vals,ind_r,ind_c,sup_centers,sup_type=create_Gamma_and_center_sup(matrice=grid_matrix,Nx=num_cel_x,Ny=num_cel_y,Nz=num_cel_z,\
                                                                      map_volumes=mapping_vols, min_v=minimum_vertex, \
                                sx=size_x,sy=size_y,sz=size_z, nodi=nodes, nodi_red=nodes_red, ext_grids=externals_grids)

    Gamma = scisparse.csc_matrix((vals, (ind_r, ind_c)), shape=(nodes_red.shape[0], sup_centers.shape[0]))

    print('Number of Surfaces (without air):', Gamma.shape[1])

    map_for_Ax, n_for_Ax = create_mapping_Ax(matrice=grid_matrix,Nx=num_cel_x,Ny=num_cel_y,Nz=num_cel_z)
    map_for_Ay, n_for_Ay = create_mapping_Ay(matrice=grid_matrix,Nx=num_cel_x,Ny=num_cel_y,Nz=num_cel_z)
    map_for_Az, n_for_Az = create_mapping_Az(matrice=grid_matrix,Nx=num_cel_x,Ny=num_cel_y,Nz=num_cel_z)

    ind_row,ind_col,vals_A, LiX, LiY, LiZ, LiX_bord,LiY_bord,LiZ_bord,\
    bars_Lp_x,bars_Lp_y,bars_Lp_z = \
        create_A_mats_volInd(matrice=grid_matrix,Nx=num_cel_x,Ny=num_cel_y,Nz=num_cel_z,mapping_Vox=mapping_vols, \
                                       mapAx=map_for_Ax, NAx=n_for_Ax, mapAy=map_for_Ay, NAy=n_for_Ay, mapAz=map_for_Az, NAz=n_for_Az,\
                                       sx=size_x,sy=size_y,sz=size_z, min_v=minimum_vertex, nodi=nodes, nodi_red=nodes_red)

    A = scisparse.csc_matrix((vals_A, (ind_row, ind_col)), shape=(n_for_Ax+n_for_Ay+n_for_Az,nodes_red.shape[0]))

    print('Edges without air:', n_for_Ax+n_for_Ay+n_for_Az)
    print('Nodes without air:', Gamma.shape[0])

    diag_R,diag_Cd=compute_diagonals(MATER=materials, sx=size_x,sy=size_y,sz=size_z, lix_mat=LiX, liy_mat=LiY, liz_mat=LiZ,\
                                     lix_border=LiX_bord, liy_border=LiY_bord, liz_border=LiZ_bord)
    

    

    return A, Gamma, port_matrix, lumped_el_matrix, sup_centers, sup_type, bars_Lp_x, bars_Lp_y, bars_Lp_z, diag_R, diag_Cd

