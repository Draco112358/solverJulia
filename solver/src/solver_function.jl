using SparseArrays
using LinearAlgebra

struct escals
    Lp
    P
    R
    Cd
    Is
    Yle
    freq
end

struct out_class
    freq 
    S
end

struct gmers_counter
    disp
    niter
end

function ver_con(A,B)
    return vcat(A,B)
end

function compute_Z_self(R,Cd,w)
    len_R=size(R)[1]
    Z_self=zeros(Complex,len_R,1)
    for cont in range(1,stop=len_R)
        for aux in range(1, stop=4)
            if R[cont,aux]!=0 && Cd[cont,aux]!=0
                Z_self[cont]=Z_self[cont]+1.0/(1.0/R[cont,aux]+1j*w*Cd[cont,aux])
            else
                if R[cont,aux]!=0
                    Z_self[cont]=Z_self[cont]+R[cont,aux]
                else
                    if Cd[cont,aux]!=0
                        Z_self[cont]=Z_self[cont]+1.0/(1j*w*Cd[cont,aux])
                    end
                end
            end
        end
    end
    return Z_self
end

function build_Yle_S(lumped_elements, ports, escalings, n, w, val_chiusura)
    l1 = size(lumped_elements.le_nodes)[1]
    l2 = size(ports.port_nodes)[1]
    Ntot = 2 * l1 + 2 * l2
    contenitore = zeros(Int64, Ntot)
    for k in range(1, stop=l1)
        contenitore[k] = lumped_elements.le_nodes[k, 1]
        contenitore[k + l1] = lumped_elements.le_nodes[k, 2]
    end
    for k in range(1, stop=l2)
        contenitore[k + 2 * l1] = ports.port_nodes[k, 1]
        contenitore[k + 2 * l1 + l2] = ports.port_nodes[k, 2]
    end

    N_ele = length(unique(contenitore))

    NNz_max = N_ele * N_ele

    ind_r = zeros(Int64, NNz_max)
    ind_c = zeros(Int64, NNz_max)
    vals = zeros(Complex, NNz_max)

    nlum = size(lumped_elements.le_nodes)[1]

    cont = 1
    for c1 in range(1, stop=nlum)
        n1 = lumped_elements.le_nodes[c1, 1]
        n2 = lumped_elements.le_nodes[c1, 2]

        ind1 = findall((ind_r == n1 && !iszero), ind_r)
        ind2 = findall((ind_c == n1 && !iszero), ind_c)
        ind = intersect(ind1, ind2)

        if length(ind) == 0
            ind_r[cont] = n1
            ind_c[cont] = n1
            if lumped_elements.type[c1] == 2
                vals[cont] = 1j * w * lumped_elements.value[c1]
            else
                vals[cont] = 1.0 / lumped_elements.value[c1]
            end
            cont = cont + 1
            
        else
            if lumped_elements.type[c1] == 2
                vals[ind[1]] = vals[ind[1]] + 1j * w * lumped_elements.value[c1]
            else
                vals[ind[1]] = vals[ind[1]] + 1.0 / lumped_elements.value[c1]
            end
        end


        ind1 = findall((ind_r == n2 && !iszero), ind_r)
        ind2 = findall((ind_c == n2 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n2, ind_r, n2)
        #ind2 = ifelse(ind_c == n2, ind_c, n2)
        ind = intersect(ind1, ind2)

        if length(ind) == 0
            ind_r[cont] = n2
            ind_c[cont] = n2
            if lumped_elements.type[c1] == 2
                vals[cont] = 1j * w * lumped_elements.value[c1]
            else
                vals[cont] = 1.0 / lumped_elements.value[c1]
            end
            cont = cont + 1
        else
            if lumped_elements.type[c1] == 2
                vals[ind[1]] = vals[ind[1]] + 1j * w * lumped_elements.value[c1]
            else
                vals[ind[1]] = vals[ind[1]] + 1.0 / lumped_elements.value[c1]
            end
        end

        ind1 = findall((ind_r == n1 && !iszero), ind_r)
        ind2 = findall((ind_c == n2 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n1, ind_r, n1)
        #ind2 = ifelse(ind_c == n2, ind_c, n2)
        ind = intersect(ind1, ind2)


        if length(ind) == 0
            if lumped_elements.type[c1] == 2
                vals[cont] = -1j * w * lumped_elements.value[c1]
            else
                vals[cont] = -1.0 / lumped_elements.value[c1]
            end
            ind_r[cont] = n1
            ind_c[cont] = n2
            cont = cont + 1
        else
            if lumped_elements.type[c1] == 2
                vals[ind[1]] = vals[ind[1]] - 1j * w * lumped_elements.value[c1]
            else
                vals[ind[1]] = vals[ind[1]] - 1.0 / lumped_elements.value[c1]
            end
        end

        ind2 = findall((ind_r == n2 && !iszero), ind_r)
        ind1 = findall((ind_c == n1 && !iszero), ind_c)
        #ind2 = ifelse(ind_r == n2, ind_r, n2)
        #ind1 = ifelse(ind_c == n1, ind_c, n1)
        ind = intersect(ind1, ind2)
        if length(ind) == 0
            if lumped_elements.type[c1] == 2
                vals[cont] = -1j * w * lumped_elements.value[c1]
            else
                vals[cont] = -1.0 / lumped_elements.value[c1]
            end
            ind_r[cont] = n2
            ind_c[cont] = n1
            cont = cont + 1
        else
            if lumped_elements.type[c1] == 2
                vals[ind[1]] = vals[ind[1]] - 1j * w * lumped_elements.value[c1]
            else
                vals[ind[1]] = vals[ind[1]] - 1.0 / lumped_elements.value[c1]
            end
        end
    end

    nport = size(ports.port_nodes)[1]

    for c1 in range(1, stop=nport)

        n1 = ports.port_nodes[c1, 1]
        n2 = ports.port_nodes[c1, 2]

        ind1 = findall((ind_r == n1 && !iszero), ind_r)
        ind2 = findall((ind_c == n1 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n1, ind_r, n1)
        #ind2 = ifelse(ind_c == n1, ind_c, n1)

        ind = intersect(ind1, ind2)

        if length(ind) == 0
            ind_r[cont] = n1
            ind_c[cont] = n1
            vals[cont] = 1.0 / val_chiusura
            cont = cont + 1
        else
            vals[ind[1]] = vals[ind[1]] + 1 / val_chiusura
        end

        ind1 = findall((ind_r == n2 && !iszero), ind_r)
        ind2 = findall((ind_c == n2 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n2, ind_r, n2)
        #ind2 = ifelse(ind_c == n2, ind_c, n2)
        ind = intersect(ind1, ind2)

        if length(ind) == 0
            ind_r[cont] = n2
            ind_c[cont] = n2
            vals[cont] = 1.0 / val_chiusura
            cont = cont + 1
        else
            vals[ind[1]] = vals[ind[1]] + 1.0 / val_chiusura
        end

        ind1 = findall((ind_r == n1 && !iszero), ind_r)
        ind2 = findall((ind_c == n2 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n1, ind_r, n1)
        #ind2 = ifelse(ind_c == n2, ind_c, n2)
        ind = intersect(ind1, ind2)

        if length(ind) == 0
            vals[cont] = -1.0 / val_chiusura
            ind_r[cont] = n1
            ind_c[cont] = n2
            cont = cont + 1
        else
            vals[ind[1]] = vals[ind[1]] - 1.0 / val_chiusura
        end

        ind1 = findall((ind_r == n2 && !iszero), ind_r)
        ind2 = findall((ind_c == n1 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n2, ind_r, n2)
        #ind2 = ifelse(ind_c == n1, ind_c, n1)
        ind = intersect(ind1, ind2)

        if ind.size == 0
            vals[cont] = -1.0 / val_chiusura
            ind_r[cont] = n2
            ind_c[cont] = n1
            cont = cont + 1
        else
            vals[ind[1]] = vals[ind[1]] - 1.0 / val_chiusura
        end
    end

    Yle = sparse(ind_r[1:cont], ind_c[1:cont], escalings.Yle*vals[1:cont])

    return Yle
end

function build_Yle_S_no_scal(lumped_elements, ports, n, w, val_chiusura)

    l1 = size(lumped_elements.le_nodes)[1]
            
    
    l2 = size(ports.port_nodes)[1]
    Ntot = 2 * l1 + 2 * l2
    contenitore = zeros(Int64, Ntot)
    for k in range(1, stop=l1)
        contenitore[k] = lumped_elements.le_nodes[k, 1]
        contenitore[k + l1] = lumped_elements.le_nodes[k, 2]
    end
    for k in range(1, stop=l2)
        contenitore[k + 2 * l1] = ports.port_nodes[k, 1]
        contenitore[k + 2 * l1 + l2] = ports.port_nodes[k, 2]
    end

    N_ele = length(unique(contenitore))

    NNz_max = N_ele * N_ele

    ind_r = zeros(Int64, NNz_max)
    ind_c = zeros(Int64, NNz_max)
    vals = zeros(Complex, NNz_max)

    nlum = size(lumped_elements.le_nodes)[1]

    cont = 1
    for c1 in range(1, stop=nlum)
        n1 = lumped_elements.le_nodes[c1, 1]
        n2 = lumped_elements.le_nodes[c1, 2]
        ind1 = findall((ind_r == n1 && !iszero), ind_r)
        ind2 = findall((ind_c == n1 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n1, ind_r, n1)
        #ind2 = ifelse(ind_c == n1, ind_c, n1)
        ind = intersect(ind1, ind2)
        if length(ind) == 0
            ind_r[cont] = n1
            ind_c[cont] = n1
            if lumped_elements.type[c1] == 2
                vals[cont] = 1j * w * lumped_elements.value[c1]
            else
                vals[cont] = 1.0 / lumped_elements.value[c1]
            cont = cont + 1
            end
        else
            if lumped_elements.type[c1] == 2
                vals[ind[1]] = vals[ind[1]] + 1j * w * lumped_elements.value[c1]
            else
                vals[ind[1]] = vals[ind[1]] + 1.0 / lumped_elements.value[c1]
            end
        end

        ind1 = findall((ind_r == n2 && !iszero), ind_r)
        ind2 = findall((ind_c == n2 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n2, ind_r, n2)
        #ind2 = ifelse(ind_c == n2, ind_c, n2)
        ind = intersect(ind1, ind2)

        if length(ind) == 0
            ind_r[cont] = n2
            ind_c[cont] = n2
            if lumped_elements.type[c1] == 2
                vals[cont] = 1j * w * lumped_elements.value[c1]
            else
                vals[cont] = 1.0 / lumped_elements.value[c1]
            cont = cont + 1
            end
        else
            if lumped_elements.type[c1] == 2
                vals[ind[1]] = vals[ind[1]] + 1j * w * lumped_elements.value[c1]
            else
                vals[ind[1]] = vals[ind[1]] + 1.0 / lumped_elements.value[c1]
            end
        end

        ind1 = findall((ind_r == n1 && !iszero), ind_r)
        ind2 = findall((ind_c == n2 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n1, ind_r, n1)
        #ind2 = ifelse(ind_c == n2, ind_c, n2)
        ind = intersect(ind1, ind2)

        if length(ind) == 0
            if lumped_elements.type[c1] == 2
                vals[cont] = -1j * w * lumped_elements.value[c1]
            else
                vals[cont] = -1.0 / lumped_elements.value[c1]
            end
            ind_r[cont] = n1
            ind_c[cont] = n2
            cont = cont + 1
        else
            if lumped_elements.type[c1] == 2
                vals[ind[1]] = vals[ind[1]] - 1j * w * lumped_elements.value[c1]
            else
                vals[ind[1]] = vals[ind[1]] - 1.0 / lumped_elements.value[c1]
            
        ind2 = findall((ind_r == n2 && !iszero), ind_r)
        ind1 = findall((ind_c == n1 && !iszero), ind_c)  
        #ind2 = ifelse(ind_r == n2, ind_r, n2)      
        #ind1 = ifelse(ind_c == n1, ind_c, n1)
        ind = intersect(ind1, ind2)
        if length(ind) == 0
            if lumped_elements.type[c1] == 2
                vals[cont] = -1j * w * lumped_elements.value[c1]
            else
                vals[cont] = -1.0 / lumped_elements.value[c1]
            end
            ind_r[cont] = n2
            ind_c[cont] = n1
            cont = cont + 1
        else
            if lumped_elements.type[c1] == 2
                vals[ind[1]] = vals[ind[1]] - 1j * w * lumped_elements.value[c1]
            else
                vals[ind[1]] = vals[ind[1]] - 1.0 / lumped_elements.value[c1]
            end
        end
    end

    nport = size(ports.port_nodes)[1]

    for c1 in range(1, stop=nport)

        n1 = ports.port_nodes[c1, 1]
        n2 = ports.port_nodes[c1, 2]

        ind1 = findall((ind_r == n1 && !iszero), ind_r)
        ind2 = findall((ind_c == n2 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n1, ind_r, n1)
        #ind2 = ifelse(ind_c == n1, ind_c, n1)

        ind = intersect(ind1, ind2)

        if length(ind) == 0
            ind_r[cont] = n1
            ind_c[cont] = n1
            vals[cont] = 1.0 / val_chiusura
            cont = cont + 1
        else
            vals[ind[1]] = vals[ind[1]] + 1 / val_chiusura
        end

        ind1 = findall((ind_r == n2 && !iszero), ind_r)
        ind2 = findall((ind_c == n2 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n2, ind_r, n2)
        #ind2 = ifelse(ind_c == n2, ind_c, n2)
        ind = intersect(ind1, ind2)

        if length(ind) == 0
            ind_r[cont] = n2
            ind_c[cont] = n2
            vals[cont] = 1.0 / val_chiusura
            cont = cont + 1
        else
            vals[ind[1]] = vals[ind[1]] + 1.0 / val_chiusura
        end
        ind1 = findall((ind_r == n1 && !iszero), ind_r)
        ind2 = findall((ind_c == n2 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n1, ind_r, n1)
        #ind2 = ifelse(ind_c == n2, ind_c, n2)
        ind = intersect(ind1, ind2)

        if length(ind) == 0
            vals[cont] = -1.0 / val_chiusura
            ind_r[cont] = n1
            ind_c[cont] = n2
            cont = cont + 1
        else
            vals[ind[1]] = vals[ind[1]] - 1.0 / val_chiusura
        end
        ind1 = findall((ind_r == n2 && !iszero), ind_r)
        ind2 = findall((ind_c == n1 && !iszero), ind_c)
        #ind1 = ifelse(ind_r == n2, ind_r, n2)
        #ind2 = ifelse(ind_c == n1, ind_c, n1)
        ind = intersect(ind1, ind2)

        if length(ind) == 0
            vals[cont] = -1.0 / val_chiusura
            ind_r[cont] = n2
            ind_c[cont] = n1
            cont = cont + 1
        else
            vals[ind[1]] = vals[ind[1]] - 1.0 / val_chiusura
        end
    end

    Yle = sparse(ind_r[0:cont], ind_c[0:cont], vals[0:cont])

    return Yle
end

function precond_3_3_vector(LU_S,invZ,invP,A,Gamma,w,X1,X2,X3)

    n1=length(X1)
    n2=length(X2)
    n3=length(X3)

    i1=range(1, stop=n1)
    i2=range(n1,stop=n1+n2)
    i3=range(n1+n2,stop=n1+n2+n3)

    Y=zeros(Complex, n1+n2+n3, 1)


    #da rivedere
    M1 = dot(sparse(invZ), X1)
    #M1 = csc_matrix.dot(invZ, X1)
    M2 = LU_S.solve(dot(sparse(transpose(A)), M1))
    #M2 = LU_S.solve(csc_matrix.dot(A.transpose(),M1))
    M3 = dot(sparse(invP), X2)
    #M3 = csc_matrix.dot(invP,X2)
    M4 = LU_S.solve(dot(sparse(Gamma), M3))
    #M4 = LU_S.solve(csc_matrix.dot(Gamma,M3))
    M5 = LU_S.solve(X3)

    Y[np.ix_(i1)] = Y[np.ix_(i1)]+M1-1.0*dot(sparse(invZ),dot(sparse(A), M2))
    Y[np.ix_(i1)] = Y[np.ix_(i1)]+1j*w*dot(sparse(invZ),dot(sparse(A), M4))
    Y[np.ix_(i1)] = Y[np.ix_(i1)]-1.0*dot(sparse(invZ),dot(sparse(A), M5))

    #Y[np.ix_(i1)] = Y[np.ix_(i1)]+M1-1.0*csc_matrix.dot(invZ,csc_matrix.dot(A,M2))
    #Y[np.ix_(i1)] = Y[np.ix_(i1)]+1j*w*csc_matrix.dot(invZ, csc_matrix.dot(A,M4))
    #Y[np.ix_(i1)] = Y[np.ix_(i1)]-1.0*csc_matrix.dot(invZ, csc_matrix.dot(A, M5))

    Y[np.ix_(i2)] = Y[np.ix_(i2)]+dot(sparse(invP),dot(sparse(transpose(Gamma)), M2))
    Y[np.ix_(i2)] = Y[np.ix_(i2)] + M3 -1j*w*dot(sparse(invP),dot(sparse(transpose(Gamma)), M4))
    Y[np.ix_(i2)] = Y[np.ix_(i2)]+dot(sparse(invP),dot(sparse(transpose(Gamma)), M5))

    # Y[np.ix_(i2)] = Y[np.ix_(i2)]+csc_matrix.dot(invP, csc_matrix.dot(Gamma.transpose(), M2))
    # Y[np.ix_(i2)] = Y[np.ix_(i2)] + M3 -1j*w*csc_matrix.dot(invP, csc_matrix.dot(Gamma.transpose(), M4))
    # Y[np.ix_(i2)] = Y[np.ix_(i2)]+csc_matrix.dot(invP, csc_matrix.dot(Gamma.transpose(), M5))

    Y[np.ix_(i3)] = Y[np.ix_(i3)]+M2
    Y[np.ix_(i3)] = Y[np.ix_(i3)]-1j*w*M4
    Y[np.ix_(i3)] = Y[np.ix_(i3)]+M5

    # Y[np.ix_(i3)] = Y[np.ix_(i3)]+M2
    # Y[np.ix_(i3)] = Y[np.ix_(i3)]-1j*w*M4
    # Y[np.ix_(i3)] = Y[np.ix_(i3)]+M5

    return Y
end

function precond_3_3_Kt(LU_S, invZ, invP, A,Gamma, n1,n2, X3)

    n3 = length(X3)

    i1 = range(1, stop=n1)
    i2 = range(n1, stop=n1 + n2)
    i3 = range(n1 + n2, stop=n1 + n2 + n3)

    Y = zeros(Complex, n1 + n2 + n3,1)

    M5 = LU_S.solve(X3)

    Y[np.ix_(i1)] = Y[np.ix_(i1)] - 1.0*csc_matrix.dot(invZ, csc_matrix.dot(A, M5))

    Y[np.ix_(i2)] = Y[np.ix_(i2)] + csc_matrix.dot(invP, csc_matrix.dot(Gamma.transpose(), M5))

    Y[np.ix_(i3)] = Y[np.ix_(i3)] + M5

    return Y
end