using SparseArrays
using LinearAlgebra
using IterativeSolvers
using LinearMaps

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

        ind1 = findall(ind_r == n1 , ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n1, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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


        ind1 = findall(ind_r == n2, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n2 , ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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

        ind1 = findall(ind_r == n1, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n2, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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

        ind2 = findall(ind_r == n2, ind_r)
        ind2 = filter(i -> !iszero(ind_r[i]), ind2)
        ind1 = findall(ind_c == n1, ind_c)
        ind1 = filter(i -> !iszero(ind_c[i]), ind1)
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

        ind1 = findall(ind_r == n1, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n1, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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

        ind1 = findall(ind_r == n2, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n2, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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

        ind1 = findall(ind_r == n1, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n2, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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

        ind1 = findall(ind_r == n2, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n1, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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
        ind1 = findall(ind_r == n1, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n1, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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

        ind1 = findall(ind_r == n2, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n2, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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

        ind1 = findall(ind_r == n1, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n2, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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
            
        ind2 = findall(ind_r == n2, ind_r)
        ind2 = filter(i -> !iszero(ind_r[i]), ind2)
        ind1 = findall(ind_c == n1, ind_c)  
        ind1 = filter(i -> !iszero(ind_c[i]), ind1)
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

        ind1 = findall(ind_r == n1, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n2, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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

        ind1 = findall(ind_r == n2, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n2, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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
        ind1 = findall(ind_r == n1, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n2, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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
        ind1 = findall(ind_r == n2, ind_r)
        ind1 = filter(i -> !iszero(ind_r[i]), ind1)
        ind2 = findall(ind_c == n1, ind_c)
        ind2 = filter(i -> !iszero(ind_c[i]), ind2)
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

    Yle = sparse(ind_r[1:cont], ind_c[1:cont], vals[1:cont])

    return Yle
end

function precond_3_3_vector(L, U, p,invZ,invP,A,Gamma,w,X1,X2,X3)

    n1=length(X1)
    n2=length(X2)
    n3=length(X3)

    i1=range(1, stop=n1)
    i2=range(n1,stop=n1+n2)
    i3=range(n1+n2,stop=n1+n2+n3)

    Y=zeros(Complex, n1+n2+n3, 1)


    #da rivedere
    M1 = *(invZ, X1)
    #M1 = csc_matrix.*(invZ, X1)
    M2 = U\(L\*(transpose(A), M1)[p])
    #M2 = LU_S.solve(csc_matrix.dot(A.transpose(),M1))
    M3 = *(invP, X2)
    #M3 = csc_matrix.*(invP,X2)
    M4 = U\(L\*(Gamma, M3)[p])
    #M4 = LU_S.solve(csc_matrix.*(Gamma,M3))
    M5 = U\(L\X3[p])

    for i in i1
        Y[i] = Y[i]+M1-1.0*(*((invZ),*((A), M2)))
        Y[i] = Y[i]+1j*w*(*((invZ),*((A), M4)))
        Y[i] = Y[i]-1.0*(*((invZ),*((A), M5)))
    end

    #Y[np.ix_(i1)] = Y[np.ix_(i1)]+M1-1.0*csc_matrix.*(invZ,csc_matrix.*(A,M2))
    #Y[np.ix_(i1)] = Y[np.ix_(i1)]+1j*w*csc_matrix.*(invZ, csc_matrix.*(A,M4))
    #Y[np.ix_(i1)] = Y[np.ix_(i1)]-1.0*csc_matrix.*(invZ, csc_matrix.*(A, M5))

    for i in i2
        Y[i] = Y[i]+(*(invP,*((transpose(Gamma)), M2)))
        Y[i] = Y[i] + M3 -1j*w*(*(invP,*(transpose(Gamma), M4)))
        Y[i] = Y[i]+(*(invP,*(transpose(Gamma), M5)))
    end

    # Y[np.ix_(i2)] = Y[np.ix_(i2)]+csc_matrix.*(invP, csc_matrix.*(Gamma.transpose(), M2))
    # Y[np.ix_(i2)] = Y[np.ix_(i2)] + M3 -1j*w*csc_matrix.*(invP, csc_matrix.*(Gamma.transpose(), M4))
    # Y[np.ix_(i2)] = Y[np.ix_(i2)]+csc_matrix.*(invP, csc_matrix.*(Gamma.transpose(), M5))

    for i in i3
        Y[i] = Y[i]+M2
        Y[i] = Y[i]-1j*w*M4
        Y[i] = Y[i]+M5
    end

    # Y[np.ix_(i3)] = Y[np.ix_(i3)]+M2
    # Y[np.ix_(i3)] = Y[np.ix_(i3)]-1j*w*M4
    # Y[np.ix_(i3)] = Y[np.ix_(i3)]+M5

    return Y
end

function precond_3_3_Kt(L, U, p, invZ, invP, A,Gamma, n1,n2, X3)

    n3 = length(X3)

    i1 = range(1, stop=n1)
    i2 = range(n1, stop=n1 + n2)
    i3 = range(n1 + n2, stop=n1 + n2 + n3)

    Y = zeros(Complex, n1 + n2 + n3,1)

    M5 = U\(L\X3[p])

    for i in i1
        Y[i] = Y[i] - 1.0*csc_matrix.*(invZ, csc_matrix.*(A, M5))
    end

    for i in i2
        Y[i] = Y[i] + csc_matrix.*(invP, csc_matrix.*(Gamma.transpose(), M5))
    end

    for i in i3
        Y[i] = Y[i] + M5
    end

    return Y
end

function s2z(S,Zo)
    num_ports=size(S)[1]
    nfreq=size(S)[3]
    Z = zeros(Complex, num_ports, num_ports, nfreq)
    Id = Matrix{Int64}(I, num_ports, num_ports)
    for cont in range(1, stop=nfreq)
        Z[:,:,cont]=Zo*((Id-1.0*S[:,:,cont])\(Id+S[:,:,cont]))
    
    end
    return Z
end

function s2y(S,Zo)
    num_ports=size(S)[1]
    nfreq=size(S)[3]
    Y = zeros(Complex, num_ports, num_ports, nfreq)
    Id = Matrix{Int64}(I, num_ports, num_ports)
    for cont in range(1, stop=nfreq)
        Y[:,:,cont]=Zo*((Id+S[:,:,cont])\(Id-1.0*S[:,:,cont]))
    end
    return Y
end

function ComputeMatrixVector(w,escalings,A,Gamma,P_mat,Lp_x_mat,Lp_y_mat,Lp_z_mat,
                        Z_self,Yle,invZ,invP,L, U, p, x)

    mx = size(Lp_x_mat)[1]
    my = size(Lp_y_mat)[1]
    mz = size(Lp_z_mat)[1]

    m = mx + my + mz
    ns = size(Gamma)[2]
    n = size(Gamma)[1]
    I = zeros(Complex, m, 1)
    Q = zeros(Complex, ns, 1)
    Phi = zeros(Complex, n, 1)
    I[1:m, 1] = x[1:m]
    Q[1:ns, 1] = x[m:m + ns]
    Phi[1:n, 1] = x[m + ns:m + ns + n]
    Y1 = zeros(Complex, m, 1)

    ia1 = range(1, stop=mx)
    ia2 = range(mx, stop=mx + my)
    ia3 = range(mx + my, stop=mx + my + mz)

    Y1[ia1] = 1j * w * escalings.Lp * *(Lp_x_mat, I[ia1])
    Y1[ia2] = 1j * w * escalings.Lp * *(Lp_y_mat, I[ia2])
    Y1[ia3] = 1j * w * escalings.Lp * *(Lp_z_mat, I[ia3])

    Y1=Y1+Z_self.*I+*((A), Phi)

    Y2 = escalings.P * np.*(P_mat,Q) -1.0*(*(transpose(Gamma), Phi))
    Y3 = -1.0*(*(transpose(A), I)) + *((Yle), Phi) +1j*w*(*(Gamma, Q))

    MatrixVector = precond_3_3_vector(L, U, p, invZ, invP, A,Gamma, w, Y1, Y2, Y3)

    return MatrixVector
end

function Quasi_static_iterative_solver(freq_in,A,Gamma,P_mat,Lp_x_mat,Lp_y_mat,Lp_z_mat,diag_R,diag_Cd,ports,lumped_elements,GMRES_settings)

    escalings = escals(1e6, 1e-12, 1e-3, 1e12, 1e3, 1e3, 1e-9)
    #escalings = escals(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    freq = freq_in * escalings.freq
    # GMRES settings - ---------------------------
    Inner_Iter = GMRES_settings.Inner_Iter
    Outer_Iter = GMRES_settings.Outer_Iter
    # -------------------------------------------

    mx = size(Lp_x_mat)[1]
    my = size(Lp_y_mat)[1]
    mz = size(Lp_z_mat)[1]

    m = mx + my + mz
    n = size(A)[2]
    ns = size(Gamma)[2]

    w = 2 * mt.pi * freq

    nfreq = size(w)[1]

    Is = zeros(Float64, n, 1)

    num_ports=size(ports.port_start)[1]

    S = zeros(Complex, num_ports, num_ports, nfreq)

    X_prec=zeros(Complex, m+n+ns, num_ports)

    diag_P = zeros(Float64, ns)
    for c in range(1, stop=ns)
        diag_P[c]=escalings.P*P_mat[c,c]
    end

    diag_Lp = zeros(Float64, m)
    for c in range(1, stop=mx)
        diag_Lp[c] = escalings.Lp*Lp_x_mat[c, c]
    end
    for c in range(1, stop=my)
        diag_Lp[c+mx] = escalings.Lp*Lp_y_mat[c, c]
    end
    for c in range(1, stop=mz)
        diag_Lp[c+mx+my] = escalings.Lp*Lp_z_mat[c, c]
    end

    invP = sparse((range(1, stop=ns), range(1, stop=ns), 1. / diag_P))

    val_chiusura = 50.0

    diag_R=escalings.R*diag_R
    diag_Cd=escalings.Cd * diag_Cd

    for k in range(1, stop=nfreq)
        print("Freq n=", k+1, " - Freq Tot=", nfreq)
        Z_self = compute_Z_self(diag_R,diag_Cd, w[k])
        Yle = build_Yle_S(lumped_elements, ports, escalings, n, w[k] / escalings.freq, val_chiusura)
        invZ = sparse(range(1, stop=m), range(1, stop=m), (1. / (Z_self[:,0] + 1j * w[k] * diag_Lp)))

        SS=Yle+(*(transpose(A),*(invZ,A))+1j*w[k]*(*(Gamma,*(invP,transpose(Gamma)))))

        L, U, p=lu(SS)
        #LU_S = linalg.spilu(SS, drop_tol=1e-6, options=dict(SymmetricMode=True))

        for c1 in range(1, stop=num_ports)
            n1 = ports.port_nodes[c1, 1]
            n2 = ports.port_nodes[c1, 2]
            Is[n1] = 1.0 * escalings.Is
            Is[n2] = -1.0 * escalings.Is

            tn = precond_3_3_Kt(L, U , p, invZ, invP, A,Gamma, m, ns, Is)

            #counter = gmres_counter()

            products_law = x -> ComputeMatrixVector( w[k], escalings, A,Gamma, P_mat,Lp_x_mat,Lp_y_mat,Lp_z_mat,Z_self, Yle, invZ,invP, L, U, p,x)

            prodts = LinearMap(products_law, n + m + ns, n + m + ns)

            V_o, info = gmres(prodts, tn, x=X_prec[:, c1], reltol=GMRES_settings.tol[k], restart=nothing, maxiter=Inner_Iter, log=true)

            # V_o, info = linalg.gmres(prodts, tn, x0=X_prec[:, c1], tol=GMRES_settings.tol[k], \
            #                               restart=None, maxiter=Inner_Iter, M=None, \
            #                               callback=counter, restrt=None, atol=None, callback_type=None)

            V = V_o
            X_prec[:, c1]=V

            Is[n1] = 0.0
            Is[n2] = 0.0

            # if info == 0
            #     print("convergence reached, number of iterations: ", counter.niter)
            # else
            #     if info > 0
            #         print("convergence not reached, number of iterations: ", counter.niter)
            #     else
            #         print("illegal input or breakdown, number of iterations: ", counter.niter)
            #     end
            # end

            for c2 in range(num_ports)
                n3 = ports.port_nodes[c2, 1]
                n4 = ports.port_nodes[c2, 2]

                if c1 == c2
                    S[c1, c2, k] = (2.0 * (V[m + ns + n3] - V[m + ns + n4]) - val_chiusura) / val_chiusura
                else
                    S[c1, c2, k] = (2.0 * (V[m + ns + n3] - V[m + ns + n4])) / val_chiusura
                end

                S[c2, c1, k] = S[c1, c2, k]
            end
        end
    end
    Z=s2z(S,val_chiusura)
    Y=s2y(S,val_chiusura)
    return Z,Y,S
end