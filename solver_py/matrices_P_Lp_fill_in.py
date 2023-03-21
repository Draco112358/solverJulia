import math as mt
import cmath as cmt
import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True, fastmath=True)
def integ_surf_surf_ortho(x1v,y1v,z1,x2v,y2,z2v):

    sol=0.0

    for c1 in range(2):
        x1 = x1v[c1]
        for c2 in range(2):
            y1 = y1v[c2]
            for c3 in range(2):
                z2 = z2v[c3]
                for c4 in range(2):
                    x2 = x2v[c4]

                    R=mt.sqrt( mt.pow(x1-x2,2)+mt.pow(y1-y2,2)+mt.pow(z1-z2,2 ) )

                    term1 = -1.0/3.0* (y1-y2)*(z1-z2)*R

                    term2 = -1.0/2.0 * (x1-x2)*abs(z1-z2)*(z1-z2)*mt.atan2( (x1-x2)*(y1-y2),( abs(z1-z2)*R))

                    term3 = -0.5 * (x1-x2)*abs(y1-y2)*(y1-y2)*mt.atan2( (x1-x2)*(z1-z2),( abs(y1-y2)*R))

                    term4 = -1.0/6.0 * mt.pow(abs(x1-x2),3)*mt.atan2( (y1-y2)*(z1-z2),( abs(x1-x2)*R))

                    if abs((x1-x2)+R)<1e-16:
                        term5 = 0.0
                    else:
                        term5 = (x1-x2)*(y1-y2)*(z1-z2)*np.real(cmt.log( np.complex( (x1-x2)+R)))

                    if abs((y1-y2)+R)<1e-16:
                        term6=0.0
                    else:
                        term6 = (0.5*mt.pow(x1-x2,2)-1.0/6.0*mt.pow(z1-z2,2))*(z1-z2)*np.real(cmt.log( np.complex( (y1-y2)+R)))

                    if abs((z1-z2)+R)<1e-16:
                        term7=0.0
                    else:
                        term7 = (0.5*mt.pow(x1-x2,2)-1.0/6.0*mt.pow(y1-y2,2))*(y1-y2)*np.real(cmt.log( np.complex( (z1-z2)+R)))

                    sol = sol + mt.pow(-1.0,c1+c2+c3+c4)*(term1+term2+term3+term4+term5+term6+term7)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def integ_surf_surf_para(x1v,y1v,z1, x2v,y2v,z2):

    sol=0.0

    for c1 in range(2):
        x1 = x1v[c1]
        for c2 in range(2):
            y1 = y1v[c2]
            for c3 in range(2):
                y2 = y2v[c3]
                for c4 in range(2):
                    x2 = x2v[c4]
                    R = mt.sqrt(mt.pow(x1 - x2, 2) + mt.pow(y1 - y2, 2) + mt.pow(z1 - z2, 2))

                    term1 = -1.0/6.0 * (mt.pow(x1 - x2, 2)+ mt.pow(y1 - y2, 2)-2.0* mt.pow(z1 - z2, 2))*R

                    term2 = -1.0*(x1-x2)*(y1-y2)*abs(z1-z2)*mt.atan2(  (x1-x2)*(y1-y2),(abs(z1-z2)*R) )

                    if abs((y1-y2)+R)<1e-16:
                        term3 = 0.0
                    else:
                        term3 = 0.5*(mt.pow(x1 - x2, 2) -mt.pow(z1 - z2, 2))*(y1-y2)*np.real(cmt.log( np.complex((y1-y2)+R)))

                    if abs((x1-x2)+R)<1e-16:
                        term4 = 0.0
                    else:
                        term4 = 0.5*(mt.pow(y1 - y2, 2) - mt.pow(z1 - z2, 2))*(x1-x2)*np.real(cmt.log( np.complex((x1-x2)+R)))

                    sol = sol + mt.pow(-1.0,c1+c2+c3+c4)*(term1+term2+term3+term4)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def integ_line_surf_para( x1v,y1v,z1,x2v,y2,z2):

    sol = 0.0

    for c1 in range(2):
        x1 = x1v[c1]
        for c2 in range(2):
            y1 = y1v[c2]
            for c3 in range(2):
                x2 = x2v[c3]

                R = mt.sqrt(mt.pow(x1 - x2, 2) + mt.pow(y1 - y2, 2) + mt.pow(z1 - z2, 2))

                if abs((x1 - x2) + R) < 1e-16:
                    term1 = 0.0
                else:
                    term1 = (x1 - x2) * (y1 - y2) * np.real(cmt.log( np.complex((x1-x2)+R)))

                if abs((y1 - y2) + R) < 1e-16:
                    term2 = 0.0
                else:
                    term2 = (mt.pow(x1 - x2, 2) - mt.pow(z1 - z2, 2)) / 2.0 *np.real(cmt.log( np.complex((y1-y2)+R)))

                term3 = -(x1 - x2) * abs(z1 - z2) * mt.atan2((x1 - x2) * (y1 - y2), (abs(z1 - z2) * R))

                term4 = -(y1 - y2) / 2.0 * R

                sol = sol + mt.pow(-1.0,c1 + c2 + c3) * (term1 + term2 + term3 + term4)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def integ_line_surf_ortho(x1v,y1,z1, x2,y2v,z2v):

    sol = 0.0

    for c1 in range(2):
        x1 = x1v[c1]
        for c2 in range(2):
            y2 = y2v[c2]
            for c3 in range(2):
                z2 = z2v[c3]

                R = mt.sqrt(mt.pow(x1 - x2, 2) + mt.pow(y1 - y2, 2) + mt.pow(z1 - z2, 2))

                if abs((x1 - x2) + R) < 1e-16:
                    term1 = 0.0
                else:
                    term1 = (y1 - y2) * (z1 - z2) * np.real(cmt.log( np.complex((x1-x2)+R)))

                if abs((y1 - y2) + R) < 1e-16:
                    term2 = 0.0
                else:
                    term2 = (x1 - x2) * (z1 - z2) * np.real(cmt.log( np.complex((y1-y2))))

                if abs((z1 - z2) + R) < 1e-16:
                    term3 = 0.0
                else:
                    term3 = (y1 - y2) * (x1 - x2) * np.real(cmt.log( np.complex((z1-z2)+R)))

                term4 = -0.5 * abs(z1 - z2) * (z1 - z2) * mt.atan2((x1 - x2) * (y1 - y2), (abs(z1 - z2) * R))

                term5 = -0.5 * abs(y1 - y2) * (y1 - y2) * mt.atan2((x1 - x2) * (z1 - z2), (abs(y1 - y2) * R))

                term6 = -0.5 * abs(x1 - x2) * (x1 - x2) * mt.atan2((y1 - y2) * (z1 - z2), (abs(x1 - x2) * R))

                sol = sol + mt.pow(-1.0,c1 + c2 + c3+1)*(term1 + term2 + term3 + term4 + term5 + term6)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def integ_line_line_ortho_xy(x1v,y1,z1,x2,y2v,z2):

    sol = 0

    for c1 in range(2):
        x1 = x1v[c1]
        for c2 in range(2):
            y2 = y2v[c2]

            R = mt.sqrt(mt.pow(x1 - x2, 2) + mt.pow(y1 - y2, 2) + mt.pow(z1 - z2, 2))

            if abs((y1 - y2) + R) < 1e-16:
                term1 = 0.0
            else:
                term1 = (x1 - x2) * np.real(cmt.log( np.complex((y1-y2)+R)))

            if abs((x1 - x2) + R) < 1e-16:
                term2 = 0.0
            else:
                term2 = (y1 - y2) * np.real(cmt.log( np.complex((x1-x2)+R)))

            term3 = -abs(z1 - z2) * mt.atan2((x1 - x2) * (y1 - y2), (abs(z1 - z2) * R))

            sol = sol + pow(-1, c1 + c2 + 1) * (term1 + term2 + term3)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def integ_line_line_parall(x1v,y1,z1,x2v,y2,z2):
    sol = 0

    for c1 in range(2):
        x1 = x1v[c1]
        for c2 in range(2):
            x2 = x2v[c2]

            R = mt.sqrt(mt.pow(x1 - x2, 2) + mt.pow(y1 - y2, 2) + mt.pow(z1 - z2, 2))

            if abs((x1 - x2) + R) < 1e-16:
                term1 = 0.0
            else:
                term1 = (x1 - x2) * np.real(cmt.log( np.complex((x1-x2)+R)))

            sol = sol + mt.pow(-1, c1 + c2 + 1) * (term1 - R)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def integ_point_sup(x1,y1,z1,x2v,y2v,z2):

    sol = 0

    for c1 in range(2):
        x2 = x2v[c1]
        for c2 in range(2):
            y2 = y2v[c2]

            R = mt.sqrt(mt.pow(x1 - x2, 2) + mt.pow(y1 - y2, 2) + mt.pow(z1 - z2, 2))

            if abs((y1 - y2) + R) < 1e-16:
                term1 = 0.0
            else:
                term1 = (x1 - x2) * np.real(cmt.log( np.complex((y1-y2)+R)))

            if abs((x1 - x2) + R) < 1e-16:
                term2 = 0.0
            else:
                term2 = (y1 - y2) * np.real(cmt.log( np.complex((x1-x2)+R)))

            term3 = -abs(z1 - z2) * mt.atan2((x1 - x2) * (y1 - y2), (abs(z1 - z2) * R))

            sol = sol + mt.pow(-1,c1 + c2) * (term1 + term2 + term3)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def integ_line_point(x1v,y3,z3,x1,y1,z1):

    x3=x1v[0]
    x4=x1v[1]

    if (x1 - x3)==0:
        x1 = x1 - 1e-8

    if (x1 - x4)==0:
        x1 = x1 + 1e-8

    #try:
      #  check = 1.0 / (x1 - x3)
   # except ZeroDivisionError:
        #x1 = x1 - 1e-8

    #try:
       # check = 1.0 / (x1 - x4)
    #except ZeroDivisionError:
       # x1 = x1 + 1e-8

    R1 = mt.sqrt(mt.pow(x1 - x3, 2) + mt.pow(y1 - y3, 2) + mt.pow(z1 - z3, 2))
    R2 = mt.sqrt(mt.pow(x1 - x4, 2) + mt.pow(y1 - y3, 2) + mt.pow(z1 - z3, 2))

    Ip = np.real(cmt.log( np.complex((x1-x3)+R1+1e-20))) - np.real(cmt.log( np.complex((x1-x4)+R2+1e-20)))

    if (mt.isnan(Ip) == True or mt.isinf(Ip) == True):
        Ip = (x1 - x3) / (1e-14*abs((x1 - x3))) * np.real(cmt.log( np.complex(abs(x1-x3)+1e-20))) \
             - (x1 - x4) / (1e-14*abs((x1 - x4))) * np.real(cmt.log( np.complex(abs(x1-x4)+1e-20)))

    return Ip

@jit(nopython=True, cache=True, fastmath=True)
def check_condition_P(eps1,eps2,eps3,eps4,Sup1,Sup2,max_d,min_R,size_dim,other_dim1,other_dim2):

    max_oth=other_dim1*other_dim2

    condS = max(max_oth, size_dim) / (min_R + 1e-15)
    condX1a = Sup1*Sup2*max_d/pow(min_R+1e-15,3)
    condX1f = size_dim/(min_R+1e-15)
    condX1b = size_dim/max_oth

    supp_dim=False
    if ((condS<eps4) and ((condX1b <= eps3 or condX1f<eps1) and condX1a<eps2)):
        supp_dim=True

    return supp_dim

@jit(nopython=True, cache=True, fastmath=True)
def Integ_sup_sup(xc1, yc1, zc1, xc2, yc2, zc2, a1,b1,c1, a2,b2,c2):

    epsilon1 = 5e-3
    epsilon2 = 1e-3
    epsilon3 = 1e-3
    epsilon4 = 3e-1

    x1v = [xc1 - a1 / 2.0, xc1 + a1 / 2.0]
    y1v = [yc1 - b1 / 2.0, yc1 + b1 / 2.0]
    z1v = [zc1 - c1 / 2.0, zc1 + c1 / 2.0]

    x2v = [xc2 - a2 / 2.0, xc2 + a2 / 2.0]
    y2v = [yc2 - b2 / 2.0, yc2 + b2 / 2.0]
    z2v = [zc2 - c2 / 2.0, zc2 + c2 / 2.0]

    sup1_xz_plane = False
    sup1_yz_plane = False

    sup2_xz_plane = False
    sup2_yz_plane = False

    if (a1 <= b1 and a1 <= c1):
        sup1_yz_plane = True
        a1 = 1.0
    else:
        if(b1 <= a1 and b1 <= c1):
            sup1_xz_plane = True
            b1 = 1.0
        else:
            c1 = 1.0

    if (a2 <= b2 and a2 <= c2):
        sup2_yz_plane = True
        a2 = 1.0
    else:
        if(b2 <= a2 and b2 <= c2):
            sup2_xz_plane = True
            b2 = 1.0
        else:
            c2 = 1.0

    sup1 = a1 * b1 * c1
    sup2 = a2 * b2 * c2

    supp_x1 = False
    supp_y1 = False
    supp_z1 = False
    supp_x2 = False
    supp_y2 = False
    supp_z2 = False

    aux_x = [abs(x1v[0] - x2v[0]), abs(x1v[0] - x2v[1]), abs(x1v[1] - x2v[0]), abs(x1v[1] - x2v[1])]
    aux_y = [abs(y1v[0] - y2v[0]), abs(y1v[0] - y2v[1]), abs(y1v[1] - y2v[0]), abs(y1v[1] - y2v[1])]
    aux_z = [abs(z1v[0] - z2v[0]), abs(z1v[0] - z2v[1]), abs(z1v[1] - z2v[0]), abs(z1v[1] - z2v[1])]

    min_R = mt.sqrt(mt.pow(min(aux_x), 2) + mt.pow(min(aux_y), 2) + mt.pow(min(aux_z), 2))

    max_d = max(aux_x)
    supp_x1 = check_condition_P(epsilon1, epsilon2, epsilon3, epsilon4, sup1, sup2, max_d, min_R, a1, b1, c1)
    supp_x2 = check_condition_P(epsilon1, epsilon2, epsilon3, epsilon4, sup1, sup2, max_d, min_R, a2, b2, c2)

    max_d = max(aux_y)
    supp_y1 = check_condition_P(epsilon1, epsilon2, epsilon3, epsilon4, sup1, sup2, max_d, min_R, b1, a1, c1)
    supp_y2 = check_condition_P(epsilon1, epsilon2, epsilon3, epsilon4, sup1, sup2, max_d, min_R, b2, a2, c2)

    max_d = max(aux_z)
    supp_z1 = check_condition_P(epsilon1, epsilon2, epsilon3, epsilon4, sup1, sup2, max_d, min_R, c1, a1, b1)
    supp_z2 = check_condition_P(epsilon1, epsilon2, epsilon3, epsilon4, sup1, sup2, max_d, min_R, c2, a2, b2)

    if (sup1_yz_plane == True):
        supp_x1 = True
    else:
        if(sup1_xz_plane == True):
            supp_y1 = True
        else:
            supp_z1 = True

    if (sup2_yz_plane == True):
        supp_x2 = True
    else:
        if (sup2_xz_plane == True):
            supp_y2 = True
        else:
            supp_z2 = True

    sum_supp = supp_x1 + supp_y1 + supp_z1 + supp_x2 + supp_y2 + supp_z2

    # point - point to use
    if sum_supp == 6:
        integ = sup1 * sup2 / mt.sqrt( mt.pow(xc1 - xc2, 2) + mt.pow(yc1 - yc2,2)  + mt.pow(zc1 - zc2,2) )
    elif sum_supp == 5: # point - line to use
        is_point_v1 = False
        if (supp_x1 + supp_y1 + supp_z1 == 3):
            is_point_v1 = True

        if is_point_v1 == True:
            if supp_x2 == 0: # line of volume 2 along x
                integ = sup1 * sup2 / a2 * integ_line_point(x2v, yc2, zc2, xc1, yc1, zc1)
            else:
                if supp_y2 == 0: # line  of volume 2 along y
                    integ = sup1 * sup2 / b2 * integ_line_point(y2v, xc2, zc2, yc1, xc1, zc1)
                else: # line of volume 2 along z
                    integ = sup1 * sup2 / c2 * integ_line_point(z2v, xc2, yc2, zc1, xc1, yc1)
        else:
            if supp_x1 == 0: # line of volume 1 along x
                integ = sup1 * sup2 / a1 * integ_line_point(x1v, yc1, zc1, xc2, yc2, zc2)
            else:
                if supp_y1 == 0: # line of volume 1 along y
                    integ = sup1 * sup2 / b1 * integ_line_point(y1v, xc1, zc1, yc2, xc2, zc2)
                else: # line of volume 1 along z
                    integ = sup1 * sup2 / c1 * integ_line_point(z1v, xc1, yc1, zc2, xc2, yc2)
    elif sum_supp == 4:  # point-surface or line-line case
        is_point_v1 = False
        if (supp_x1 + supp_y1 + supp_z1 == 3):
            is_point_v1 = True

        is_point_v2 = False
        if (supp_x2 + supp_y2 + supp_z2 == 3):
            is_point_v2 = True

        if is_point_v1 == True: #point-surface case
            if supp_x2 == True: #surface of volume 2 in yz plane
                integ = sup1 * a2 * integ_point_sup(zc1, yc1, xc1, z2v, y2v, xc2)
            else:
                if supp_y2 == True: #surface of volume 2 in xz plane
                    integ = sup1 * b2 * integ_point_sup(xc1, zc1, yc1, x2v, z2v, yc2)
                else:
                    integ = sup1 * c2 * integ_point_sup(xc1, yc1, zc1, x2v, y2v, zc2)
        else:
            if is_point_v2 == True: #point-surface case
                if supp_x1 == True: #surface of volume 1 in yz plane
                    integ = sup2 * a1 * integ_point_sup(zc2, yc2, xc2, z1v, y1v, xc1)
                else:
                    if supp_y1== True: #surface of volume 1 in xz plane
                        integ = sup2 * b1 * integ_point_sup(xc2, zc2, yc2, x1v, z1v, yc1)
                    else: #surface of volume 1 in xy plane
                        integ=sup2*c1*integ_point_sup(xc2,yc2,zc2,x1v,y1v,zc1)
            else: #line-line case
                if supp_y1 == True and supp_z1 == True:
                    if supp_y2 == True and supp_z2 == True: #parallel lines
                        integ = b1 * c1 * b2 * c2 * integ_line_line_parall(x1v, yc1, zc1,x2v, yc2, zc2)
                    else:
                        if supp_x2 == True and supp_z2 == True: #orthogonal lines
                            integ = b1 * c1 * a2 * c2 * integ_line_line_ortho_xy(x1v, yc1, zc1, xc2, y2v, zc2)
                        else:
                            integ = b1 * c1 * a2 * b2 * integ_line_line_ortho_xy(x1v, zc1, yc1, xc2, z2v, yc2)
                else:
                    if supp_x1 == True and supp_z1 == True:
                        if supp_x2 == True and supp_z2 == True:  # parallel lines
                            integ = a1 * c1 * a2 * c2 * integ_line_line_parall(y1v, xc1, zc1,y2v, xc2, zc2)
                        else:
                            if supp_x2 == True and supp_y2 == True:  # orthogonal lines
                                integ = a1 * c1 * a2 * b2 * integ_line_line_ortho_xy(y1v, zc1, xc1, yc2, z2v, xc2)
                            else:
                                integ = a1 * c1 * b2 * c2 * integ_line_line_ortho_xy(y1v, xc1, zc1, yc2, x2v, zc2)
                    else:
                        if supp_x2 == True and supp_y2 == True:  # parallel lines
                            integ = a1 * b1 * a2 * b2 * integ_line_line_parall(z1v, xc1, yc1, z2v, xc2, yc2)
                        else:
                            if supp_x2 == True and supp_z2 == True:  # orthogonal lines
                                integ = a1 * b1 * a2 * c2 * integ_line_line_ortho_xy(z1v, yc1, xc1, zc2, y2v, xc2)
                            else:
                                integ = a1 * b1 * b2 * c2 * integ_line_line_ortho_xy(z1v, xc1, yc1, zc2, x2v, yc2)
    elif sum_supp == 3: #surface-line
        is_surf_v1 = False
        if (supp_x1 + supp_y1 + supp_z1 == 1):
            is_surf_v1 = True
        #line-surface case
        if is_surf_v1 == True: #bar1 is a surface
            if supp_x1 == True: #bar1 is a surface in y-z plane
                if supp_x2 == False: #bar2 is a line along x
                    integ = a1 * b2 * c2 * integ_line_surf_ortho(x2v, yc2, zc2, xc1, y1v, z1v)
                else:
                    if supp_y2 == False: #bar2 is a line along y
                        integ = a1 * a2 * c2 * integ_line_surf_para(y1v, z1v, xc1, y2v, zc2, xc2)
                    else: #bar2 is a line along z
                        integ = a1 * a2 * b2 * integ_line_surf_para(z1v, y1v, xc1, z2v, yc2, xc2)
            else:
                if supp_y1 == True: #bar1 is a surface in x-z plane
                    if supp_x2 == False: #bar2 is a line along x
                        integ = b1 * b2 * c2 * integ_line_surf_para(x1v, z1v, yc1, x2v, zc2, yc2)
                    else:
                        if supp_y2 == False: #bar2 is a line along y
                            integ = b1 * a2 * c2 * integ_line_surf_ortho(y2v, xc2, zc2, yc1, x1v, z1v)
                        else: #bar2 is a line along z
                            integ = b1 * a2 * b2 * integ_line_surf_para(z1v, x1v, yc1, z2v, xc2, yc2)
                else:
                    if supp_x2 == False: #bar2 is a line along x
                        integ = c1 * b2 * c2 * integ_line_surf_para(x1v, y1v, zc1, x2v, yc2, zc2)
                    else:
                        if supp_y2 == False: #bar2 is a line along y
                            integ = c1 * a2 * c2 * integ_line_surf_para(y1v, x1v, zc1, y2v, xc2, zc2)
                        else: #bar2 is a line along z
                            integ = c1 * a2 * b2 * integ_line_surf_ortho(z2v, xc2, yc2, zc1, x1v, y1v)
        else: #bar2 is a surface
            if supp_x2 == True: #bar2 is a surface in y-z plane
                if supp_x1 == False: #bar1 is a line along x
                    integ=a2*b1*c1 * integ_line_surf_ortho(x1v,yc1,zc1, xc2,y2v,z2v)
                else:
                    if supp_y1 == False: #bar1 is a line along y
                        integ=a2*a1*c1 *integ_line_surf_para( y2v,z2v,xc2,y1v,zc1,xc1)
                    else: #bar1 is a line along z
                        integ=a2*a1*b1 *integ_line_surf_para( z2v,y2v,xc2,z1v,yc1,xc1)
            else:
                if supp_y2 == True: #bar2 is a surface in x-z plane
                    if supp_x1 == False: #bar1 is a line along x
                        integ=b2*b1*c1 * integ_line_surf_para( x2v,z2v,yc2,x1v,zc1,yc1)
                    else:
                        if supp_y1 == False: #bar1 is a line along y
                            integ=b2*a1*c1 *integ_line_surf_ortho(y1v,xc1,zc1, yc2,x2v,z2v)
                        else: #bar1 is a line along z
                            integ=b2*a1*b1 *integ_line_surf_para( z2v,x2v,yc2,z1v,xc1,yc1)
                else:
                    if supp_x1 == False: #bar1 is a line along x
                        integ=c2*b1*c1 * integ_line_surf_para( x2v,y2v,zc2,x1v,yc1,zc1)
                    else:
                        if supp_y1 == False: #bar1 is a line along y
                            integ=c2*a1*c1 *integ_line_surf_para( y2v,x2v,zc2,y1v,xc1,zc1)
                        else: #bar1 is a line along z
                            integ=c2*a1*b1 *integ_line_surf_ortho(z1v,xc1,yc1, zc2,x2v,y2v)
    else: #surface-surface case
        if supp_x1 == True: #bar1 is a surface in yz plane
            if supp_x2 == True: #bar2 is a surface in yz plane
                integ = a1 * a2 * integ_surf_surf_para(y1v, z1v, xc1, y2v, z2v, xc2)
            else:
                if supp_y2 == True: #bar2 is a surface in xz plane
                    integ = a1 * b2 * integ_surf_surf_ortho(z1v, y1v, xc1, z2v, yc2, x2v)
                else: #bar2 is a surface in xy plane
                    integ = a1 * c2 * integ_surf_surf_ortho(y1v, z1v, xc1, y2v, zc2, x2v)
        else:
            if supp_y1 == True: #%bar1 is a surface in xz plane
                if supp_x2 == True: #bar2 is a surface in yz plane
                    integ = b1 * a2 * integ_surf_surf_ortho(z1v, x1v, yc1, z2v, xc2, y2v)
                else:
                    if supp_y2 == True: #bar2 is a surface in xz plane
                        integ = b1 * b2 * integ_surf_surf_para(x1v, z1v, yc1, x2v, z2v, yc2)
                    else: #bar2 is a surface in xy plane
                        integ = b1 * c2 * integ_surf_surf_ortho(x1v, z1v, yc1, x2v, zc2, y2v)
            else: #bar1 is a surface in xy plane
                if supp_x2 == True: #bar2 is a surface in yz plane
                    integ = c1 * a2 * integ_surf_surf_ortho(y1v, x1v, zc1, y2v, xc2, z2v)
                else:
                    if supp_y2 == True: #bar2 is a surface in xz plane
                        integ = c1 * b2 * integ_surf_surf_ortho(x1v, y1v, zc1, x2v, yc2, z2v)
                    else: #bar2 is a surface in xy plane
                        integ = c1 * c2 * integ_surf_surf_para(x1v, y1v, zc1, x2v, y2v, zc2)
    res=integ.real
    return res

#@jit(nopython=True, cache=True, parallel=True, fastmath=True)
@jit(nopython=True, cache=True, fastmath=True)
def compute_P_matrix(centers,sup_type,sx,sy,sz):

    eps0=8.854187816997944e-12

    N=centers.shape[0]
    P_mat=np.zeros((N, N), dtype='double')

    for m in range(N):
        if sup_type[m] == 1:
            A1 = sx * sz
            a1 = sx
            b1 = 0.0
            c1 = sz
        elif sup_type[m] == 2:
            A1 = sy * sz
            a1 = 0.0
            b1 = sy
            c1 = sz
        elif sup_type[m] == 3:
            A1 = sx * sy
            a1 = sx
            b1 = sy
            c1 = 0.0
        for n in range(m,N):
            if sup_type[n] == 1:
                A2 = sx * sz
                a2 = sx
                b2 = 0.0
                c2 = sz
            elif sup_type[n] == 2:
                A2 = sy * sz
                a2 = 0.0
                b2 = sy
                c2 = sz
            elif sup_type[n] == 3:
                A2 = sx * sy
                a2 = sx
                b2 = sy
                c2 = 0.0

            P_mat[m,n]=1.0 / (4.0 * mt.pi * eps0 * A1 * A2) * Integ_sup_sup(centers[m,0], centers[m,1], 
                                                                            centers[m,2], centers[n,0], centers[n,1], centers[n,2],
                                                                            a1, b1, c1, a2, b2, c2)
            P_mat[n, m]=P_mat[m,n]

    return P_mat

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_P_matrix_parallel(centers,sup_type,sx,sy,sz):
    eps0=8.854187816997944e-12
    N=centers.shape[0]
    P_mat=np.zeros((N, N), dtype='double')
    for (m) in prange(N):
        if sup_type[m] == 1:
            A1 = sx * sz
            a1 = sx
            b1 = 0.0
            c1 = sz
        elif sup_type[m] == 2:
            A1 = sy * sz
            a1 = 0.0
            b1 = sy
            c1 = sz
        elif sup_type[m] == 3:
            A1 = sx * sy
            a1 = sx
            b1 = sy
            c1 = 0.0
        for n in range(m,N):
            if sup_type[n] == 1:
                A2 = sx * sz
                a2 = sx
                b2 = 0.0
                c2 = sz
            elif sup_type[n] == 2:
                A2 = sy * sz
                a2 = 0.0
                b2 = sy
                c2 = sz
            elif sup_type[n] == 3:
                A2 = sx * sy
                a2 = sx
                b2 = sy
                c2 = 0.0

            P_mat[m,n]=1.0 / (4.0 * mt.pi * eps0 * A1 * A2) * Integ_sup_sup(centers[m,0], centers[m,1], 
                                                                            centers[m,2], centers[n,0], centers[n,1], centers[n,2],
                                                                            a1, b1, c1, a2, b2, c2)
            P_mat[n, m]=P_mat[m,n]

    return P_mat


@jit(nopython=True, cache=True, fastmath=True)
def compute_Lp_self(l,W,T):
    #fast Henry
    w=W/l
    t=T/l
    r = mt.sqrt(w * w + t * t)
    aw = mt.sqrt(w * w + 1.0)
    at = mt.sqrt(t * t + 1.0)
    ar = mt.sqrt(w * w + t * t + 1.0)

    mu0 = 4.0 * mt.pi * 1e-7

    Lp_Self_Rect = 2.0 * mu0 * l / mt.pi * ( 0.25 * (1 / w * mt.asinh(w / at) + \
                    1 / t * mt.asinh(t / aw) + mt.asinh(1.0 / r)) + \
                1.0 / 24.0 * (t * t / w * mt.asinh(w / (t * at * (r + ar))) \
                              + w * w / t * mt.asinh(t / (w * aw * (r + ar))) + \
                              t * t / (w * w) * mt.asinh(w * w / (t * r * (at + ar))) + \
                              w * w / (t * t) * mt.asinh(\
                    t *t / (w * r * (aw + ar))) + \
                              1.0 / (w * t * t) * mt.asinh(w * t * t / (at * (aw + ar))) + 1.0 / (\
                                          t * w * w) * mt.asinh(t * w * w / (aw * (at + ar))) \
                              ) \
                - 1.0 / 6.0 * (\
                            1.0 / (w * t) * mt.atan(w * t / ar) + t / w * mt.atan(w / (t * ar)) \
                            + w / t * mt.atan(t / (w * ar))) \
                - 1.0 / 60.0 * ((ar + r + t + at) * t * t / ((ar + r) * (r + t) * (t + at) * (at + ar)) + \
                                (ar + r + w + aw) * w * w / ((ar + r) * (r + w) * (w + aw) * (aw + ar)) + \
                                (ar + aw + 1.0 + at) / ((ar + aw) * (aw + 1.0) * (1.0 + at) * (at + ar)) \
                                ) \
                - 1.0 / 20.0 * (1.0 / (r + ar) + 1.0 / (aw + ar) + 1.0 / (at + ar)) \
                )
    return Lp_Self_Rect

@jit(nopython=True, cache=True, fastmath=True)
def integ_vol_vol(x1v,y1v,z1v, x2v,y2v,z2v):

    sol = 0.0

    for c1 in range(2):
        x1 = x1v[c1]
        for c2 in range(2):
            y1 = y1v[c2]
            for c3 in range(2):
                z1 = z1v[c3]
                for c4 in range(2):
                    x2 = x2v[c4]
                    for c5 in range(2):
                        y2 = y2v[c5]
                        for c6 in range(2):
                            z2 = z2v[c6]

                            R = mt.sqrt(mt.pow(x1 - x2, 2) + mt.pow(y1 - y2, 2) + mt.pow(z1 - z2, 2))

                            term1 = (mt.pow(x1 - x2, 4) + mt.pow(y1 - y2, 4) + mt.pow(z1 - z2, 4)\
                                     - 3.0 * mt.pow(x1 - x2, 2) * mt.pow(y1 - y2, 2) \
                                     - 3.0 * mt.pow(y1 - y2, 2) * mt.pow(z1 - z2, 2) \
                                     - 3.0 * mt.pow(x1 - x2, 2) * mt.pow(z1 - z2, 2)) * R / 60.0

                            if abs((x1 - x2) + R) < 1e-16:
                                term2 = 0.0
                            else:
                                term2 = (mt.pow(y1 - y2, 2) * mt.pow(z1 - z2, 2) / 4.0 - mt.pow(y1 - y2, 4) / 24.0 \
                                     - mt.pow(z1 - z2, 4) / 24.0) * \
                                    (x1 - x2) * np.real(cmt.log( np.complex((x1-x2)+R)))

                            if abs((z1 - z2) + R) < 1e-16:
                                term3 = 0.0
                            else:
                                term3 = (mt.pow(y1 - y2, 2) * mt.pow(x1 - x2, 2) / 4.0 - mt.pow(y1 - y2, 4) / 24.0 \
                                    - mt.pow(x1 - x2,4) / 24.0) * (z1 - z2) * np.real(cmt.log( np.complex((z1-z2)+R)))

                            if abs((y1 - y2) + R) < 1e-16:
                                term4 = 0.0
                            else:
                                term4 = (mt.pow(z1 - z2, 2) * mt.pow(x1 - x2, 2) / 4.0 - mt.pow(z1 - z2, 4)  / 24.0 \
                                     - mt.pow(x1 - x2, 4) / 24.0) * ( y1 - y2) * np.real(cmt.log( np.complex((y1-y2)+R)))

                            term5 = - abs(mt.pow(x1 - x2, 3))  * (y1 - y2) * (z1 - z2) / 6.0 \
                                    * mt.atan2((y1 - y2) * (z1 - z2),(abs(x1 - x2) * R))

                            term6 = - (x1 - x2) * abs(mt.pow(y1 - y2, 3)) * (z1 - z2) / 6.0 \
                                    * mt.atan2((x1 - x2) * (z1 - z2),(abs(y1 - y2) * R))

                            term7 = - (x1 - x2) * (y1 - y2) * abs(mt.pow(z1 - z2, 3)) / 6.0 \
                                    * mt.atan2((x1 - x2) * (y1 - y2), (abs(z1 - z2) * R))

                            sol = sol + pow(-1.0, c1 + c2 + c3 + c4 + c5 + c6 + 1) * ( \
                                        term1 + term2 + term3 + term4 + term5 + term6 + term7)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def integ_vol_surf( x1v,y1v,z1v, x2v,y2v,z2):
    sol = 0.0

    for c1 in range(2):
        x1 = x1v[c1]
        for c2 in range(2):
            y1 = y1v[c2]
            for c3 in range(2):
                z1 = z1v[c3]
                for c4 in range(2):
                    x2 = x2v[c4]
                    for c5 in range(2):
                        y2 = y2v[c5]
                        R = mt.sqrt(mt.pow(x1 - x2, 2) + mt.pow(y1 - y2, 2) + mt.pow(z1 - z2, 2))

                        term1 = (mt.pow(z1 - z2,2) / 12.0 - (mt.pow(x1 - x2,2) + mt.pow(y1 - y2, 2)) / 8.0) \
                                * (z1 - z2) * R

                        if abs((x1 - x2) + R) < 1e-16:
                            term2 = 0.0
                        else:
                            term2 = (mt.pow(y1 - y2, 2) / 2.0 - mt.pow(z1 - z2, 2) / 6.0) * (x1 - x2) * (z1 - z2) \
                                * np.real(cmt.log( np.complex((x1-x2)+R)))

                        if abs((y1 - y2) + R) < 1e-16:
                            term3 = 0.0
                        else:
                            term3 = (mt.pow(x1 - x2, 2) / 2.0 - mt.pow(z1 - z2, 2) / 6.0) * (y1 - y2) * (z1 - z2) * \
                                np.real(cmt.log(np.complex((y1 - y2) + R)))

                        if abs((z1 - z2) + R) < 1e-16:
                            term4 = 0.0
                        else:
                            term4 = (- mt.pow(x1 - x2, 4) / 24.0 - mt.pow(y1 - y2, 4) / 24.0 + \
                                 ( mt.pow(y1 - y2, 2) * mt.pow(x1 - x2, 2) ) / 4.0) * \
                                np.real(cmt.log(np.complex((z1 - z2) + R)))

                        if (mt.isnan(term4) == True or mt.isinf(term4) == True):
                            term4 = 0.0

                        term5 = - abs(mt.pow(x1 - x2, 3)) * (y1 - y2) / 6.0 * mt.atan2((y1 - y2) * (z1 - z2), (abs(x1 - x2) * R))

                        term6 = - (x1 - x2) * abs(mt.pow(y1 - y2, 3)) / 6.0 * mt.atan2((x1 - x2) * (z1 - z2), (abs(y1 - y2) * R))

                        term7 = - (x1 - x2) * (y1 - y2) * abs(z1 - z2) * (z1 - z2) / 2.0 * mt.atan2((x1 - x2) * (y1 - y2), \
                                                                                               (abs(z1 - z2) * R))

                        sol = sol + mt.pow(-1, c1 + c2 + c3 + c4 + c5 + 1) * ( \
                                    term1 + term2 + term3 + term4 + term5 + term6 + term7)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def integ_line_vol(x1v,y1v,z1v,x2v,y2,z2):

    sol = 0.0

    for c1 in range(2):
        x1 = x1v[c1]
        for c2 in range(2):
            y1 = y1v[c2]
            for c3 in range(2):
                z1 = z1v[c3]
                for c4 in range(2):
                    x2 = x2v[c4]
                    R = mt.sqrt(mt.pow(x1 - x2, 2) + mt.pow(y1 - y2, 2) + mt.pow(z1 - z2, 2))

                    term1 = -1.0 / 3.0 * (y1 - y2) * (z1 - z2) * R

                    term2 = -0.5 * (x1 - x2) * abs(z1 - z2) * (z1 - z2) * mt.atan2((x1 - x2) * (y1 - y2), \
                                                                                  (abs(z1 - z2) * R))

                    term3 = -0.5 * (x1 - x2) * abs(y1 - y2) * (y1 - y2) * mt.atan2((x1 - x2) * (z1 - z2), \
                                                                                  (abs(y1 - y2) * R))

                    term4 = -1.0 / 6.0 * abs(mt.pow(x1 - x2, 3)) * mt.atan2((y1 - y2) * (z1 - z2), (abs(x1 - x2) * R))

                    if abs((x1 - x2) + R) < 1e-16:
                        term5 = 0.0
                    else:
                        term5 = (x1 - x2) * (y1 - y2) * (z1 - z2) * np.real(cmt.log( np.complex((x1-x2)+R)))

                    if abs((y1 - y2) + R) < 1e-16:
                        term6 = 0.0
                    else:
                        term6 = (0.5 * mt.pow(x1 - x2,2) - 1.0 / 6.0 * mt.pow(z1 - z2,2)) * (z1 - z2) * \
                            np.real(cmt.log( np.complex((y1-y2)+R)))

                    if abs((z1 - z2) + R) < 1e-16:
                        term7 = 0.0
                    else:
                        term7 = (0.5 * mt.pow(x1 - x2,2) - 1.0 / 6.0 * mt.pow(y1 - y2, 2) ) * (y1 - y2) * \
                            np.real(cmt.log(np.complex((z1 - z2) + R)))


                    sol = sol + mt.pow(-1.0, c1 + c2 + c3 + c4 + 1) * ( \
                                term1 + term2 + term3 + term4 + term5 + term6 + term7)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def integ_point_vol(x1,y1,z1, x2v,y2v,z2v):

    sol = 0.0

    for c1 in range(2):
        x2 = x2v[c1]
        for c2 in range(2):
            y2 = y2v[c2]
            for c3 in range(2):
                z2 = z2v[c3]
                R = mt.sqrt(mt.pow(x1 - x2, 2) + mt.pow(y1 - y2, 2) + mt.pow(z1 - z2, 2))

                if abs((x1 - x2) + R) < 1e-16:
                    term1 = 0.0
                else:
                    term1 = (y1 - y2) * (z1 - z2) * np.real(cmt.log(np.complex((x1 - x2) + R)))

                if abs((y1 - y2) + R) < 1e-16:
                    term2 = 0.0
                else:
                    term2 = (x1 - x2) * (z1 - z2) * np.real(cmt.log(np.complex((y1 - y2) + R)))

                if abs((z1 - z2) + R) < 1e-16:
                    term3 = 0.0
                else:
                    term3 = (y1 - y2) * (x1 - x2) * np.real(cmt.log(np.complex((z1 - z2) + R)))


                term4 = -0.5 * abs(z1 - z2) * (z1 - z2) * mt.atan2((x1 - x2) * (y1 - y2), (abs(z1 - z2) * R))

                term5 = -0.5 * abs(y1 - y2) * (y1 - y2) * mt.atan2((x1 - x2) * (z1 - z2), (abs(y1 - y2) * R))

                term6 = -0.5 * abs(x1 - x2) * (x1 - x2) * mt.atan2((y1 - y2) * (z1 - z2), (abs(x1 - x2) * R))

                sol = sol + mt.pow(-1.0, c1 + c2 + c3) * (term1 + term2 + term3 + term4 + term5 + term6)

    return sol

@jit(nopython=True, cache=True, fastmath=True)
def check_condition(eps1,eps2,eps3,V1,V2,max_d,min_R,size_dim,other_dim1,other_dim2):

    max_oth=max([other_dim1,other_dim2])

    condX1a = V1*V2*max_d/pow(min_R+1e-15,3)
    condX1f = size_dim/(min_R+1e-15)
    condX1b = size_dim/max_oth

    supp_dim=False
    if ((condX1b <= eps3 or condX1f < eps1) and condX1a < eps2):
        supp_dim=True

    return supp_dim

@jit(nopython=True, cache=True, fastmath=True)
def Integ_vol_vol(x1v,y1v,z1v, x2v,y2v,z2v):

    epsilon1=5e-3
    epsilon2=1e-3
    epsilon3=1e-3
    epsilon4=3e-1

    xc1=0.5*(x1v[0]+x1v[1])
    yc1=0.5*(y1v[0]+y1v[1])
    zc1=0.5*(z1v[0]+z1v[1])
    xc2=0.5*(x2v[0]+x2v[1])
    yc2=0.5*(y2v[0]+y2v[1])
    zc2=0.5*(z2v[0]+z2v[1])

    a1=abs(x1v[1]-x1v[0])
    b1=abs(y1v[1]-y1v[0])
    c1=abs(z1v[1]-z1v[0])

    a2=abs(x2v[1]-x2v[0])
    b2=abs(y2v[1]-y2v[0])
    c2=abs(z2v[1]-z2v[0])

    V1=a1*b1*c1
    V2=a2*b2*c2

    supp_x1=False
    supp_y1=False
    supp_z1=False
    supp_x2=False
    supp_y2=False
    supp_z2=False

    aux_x = [abs(x1v[0] - x2v[0]), abs(x1v[0] - x2v[1]), abs(x1v[1] - x2v[0]), abs(x1v[1] - x2v[1])]
    aux_y = [abs(y1v[0] - y2v[0]), abs(y1v[0] - y2v[1]), abs(y1v[1] - y2v[0]), abs(y1v[1] - y2v[1])]
    aux_z = [abs(z1v[0] - z2v[0]), abs(z1v[0] - z2v[1]), abs(z1v[1] - z2v[0]), abs(z1v[1] - z2v[1])]

    min_R = mt.sqrt(mt.pow(min(aux_x), 2) + mt.pow(min(aux_y), 2) + mt.pow(min(aux_z), 2))

    if (a1 <= b1 and a1 <= c1):
        max_d = max(aux_x)
        supp_x1 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, a1, b1, c1)
        if supp_x1 == True:
            max_ed = max(b1, c1)
            if (max_ed / (min_R + 1e-15) < epsilon4):
                max_d = max(aux_y)
                supp_y1 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, b1, a1, c1)
                max_d = max(aux_z)
                supp_z1 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, c1, a1, b1)
    else:
        if (b1 <= a1 and b1 <= c1):
            max_d = max(aux_y)
            supp_y1 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, b1, a1, c1)
            if supp_y1 == True:
                max_ed = max(a1, c1)
                if (max_ed / (min_R + 1e-15) < epsilon4):
                    max_d = max(aux_x)
                    supp_x1 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, a1, b1, c1)
                    max_d = max(aux_z)
                    supp_z1 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, c1, a1, b1)
        else:
            max_d = max(aux_z)
            supp_z1 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, c1, a1, b1)
            if supp_z1 == True:
                max_ed = max(a1, b1)
                if (max_ed / (min_R + 1e-15) < epsilon4):
                    max_d = max(aux_x)
                    supp_x1 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, a1, b1, c1)
                    max_d = max(aux_y)
                    supp_y1 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, b1, a1, c1)

    if (a2 <= b2 and a2 <= c2):
        max_d = max(aux_x)
        supp_x2 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, a2, b2, c2)
        if supp_x2 == True:
            max_ed = max(b2, c2)
            if (max_ed / (min_R + 1e-15) < epsilon4):
                max_d = max(aux_y)
                supp_y2 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, b2, a2, c2)
                max_d = max(aux_z)
                supp_z2 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, c2, a2, b2)
    else:
        if (b2 <= a2 and b2 <= c2):
            max_d = max(aux_y)
            supp_y2 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, b2, a2, c2)
            if supp_y2 == True:
                max_ed = max(a2, c2)
                if (max_ed / (min_R + 1e-15) < epsilon4):
                    max_d = max(aux_x)
                    supp_x2 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, a2, b2, c2)
                    max_d = max(aux_z)
                    supp_z2 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, c2, a2, b2)
        else:
            max_d = max(aux_z)
            supp_z2 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, c2, a2, b2)
            if supp_z2 == True:
                max_ed = max(a2, b2)
                if (max_ed / (min_R + 1e-15) < epsilon4):
                    max_d = max(aux_x)
                    supp_x2 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, a2, b2, c2)
                    max_d = max(aux_y)
                    supp_y2 = check_condition(epsilon1, epsilon2, epsilon3, V1, V2, max_d, min_R, b2, a2, c2)

    sum_supp = supp_x1 + supp_y1 + supp_z1 + supp_x2 + supp_y2 + supp_z2

    # point - point to use
    if sum_supp == 6:
        integ = V1 * V2 / mt.sqrt( mt.pow(xc1 - xc2, 2) + mt.pow(yc1 - yc2, 2) + mt.pow(zc1 - zc2,2) )
    elif sum_supp == 5:  #point-line to use
        is_point_v1 = False
        if (supp_x1 + supp_y1 + supp_z1 == 3):
            is_point_v1 = True

        if is_point_v1 == True:
            if supp_x2 == False: # line of volume 2 along x
                integ = V1 * V2 / a2 * integ_line_point(x2v, yc2, zc2, xc1, yc1, zc1)
            else:
                if supp_y2 == False: #line of volume 2 along y
                    integ = V1 * V2 / b2 * integ_line_point(y2v, xc2, zc2, yc1, xc1, zc1)
                else: # line of volume 2 along z
                    integ = V1 * V2 / c2 * integ_line_point(z2v, xc2, yc2, zc1, xc1, yc1)
        else:
            if supp_x1 == False: # line of volume 1 along x
                integ = V1 * V2 / a1 * integ_line_point(x1v, yc1, zc1, xc2, yc2, zc2)
            else:
                if supp_y1 == False: # line of volume 1 along y
                    integ = V1 * V2 / b1 * integ_line_point(y1v, xc1, zc1, yc2, xc2, zc2)
                else: # %line of volume 1 along z
                    integ = V1 * V2 / c1 * integ_line_point(z1v, xc1, yc1, zc2, xc2, yc2)
    elif sum_supp == 4: #point-surface or line-line case
        is_point_v1 = False
        if (supp_x1 + supp_y1 + supp_z1 == 3):
            is_point_v1 = True
        is_point_v2 = False
        if (supp_x2 + supp_y2 + supp_z2 == 3):
            is_point_v2 = True

        if is_point_v1 == True: # point - surface case
            if supp_x2 == True: # surface of volume 2 in yz plane
                integ = V1 * a2 * integ_point_sup(zc1, yc1, xc1, z2v, y2v, xc2)
            else:
                if supp_y2 == True: #surface of volume 2 in xz plane
                    integ = V1 * b2 * integ_point_sup(xc1, zc1, yc1, x2v, z2v, yc2)
                else : #surface of volume 2 in xy plane
                    integ = V1 * c2 * integ_point_sup(xc1, yc1, zc1, x2v, y2v, zc2)
        else:
            if is_point_v2 == True: #point-surface case
                if supp_x1 == True: #surface of volume 1 in yz plane
                    integ = V2 * a1 * integ_point_sup(zc2, yc2, xc2, z1v, y1v, xc1)
                else:
                    if supp_y1 == True: #surface of volume 1 in xz plane
                        integ = V2 * b1 * integ_point_sup(xc2, zc2, yc2, x1v, z1v, yc1)
                    else: #surface of volume 1 in xy plane
                        integ = V2 * c1 * integ_point_sup(xc2, yc2, zc2, x1v, y1v, zc1)
            else: #line-line case
                if supp_y1 == True and supp_z1 == True:
                    if supp_y2 == True and supp_z2 == True: # parallel lines
                        integ = b1 * c1 * b2 * c2 * integ_line_line_parall(x1v, yc1, zc1, x2v, yc2, zc2)
                    else:
                        if supp_x2 == True and supp_z2 == True: # orthogonal lines
                            integ = b1 * c1 * a2 * c2 * integ_line_line_ortho_xy(x1v, yc1, zc1, xc2, y2v, zc2)
                        else:
                            integ = b1 * c1 * a2 * b2 * integ_line_line_ortho_xy(x1v, zc1, yc1, xc2, z2v, yc2)
                else:
                    if supp_x1 == True and supp_z1 == True:
                        if supp_x2 == True and supp_z2 == True: #parallel lines
                            integ = a1 * c1 * a2 * c2 * integ_line_line_parall(y1v, xc1, zc1, y2v, xc2, zc2)
                        else:
                            if supp_x2 == True and supp_y2 == True: # orthogonal lines
                                integ = a1 * c1 * a2 * b2 * integ_line_line_ortho_xy(y1v, zc1, xc1, yc2, z2v, xc2)
                            else:
                                integ = a1 * c1 * b2 * c2 * integ_line_line_ortho_xy(y1v, xc1, zc1, yc2, x2v, zc2)
                    else:
                        if supp_x2 == True and supp_y2 == True: # parallel lines
                            integ = a1 * b1 * a2 * b2 * integ_line_line_parall(z1v, xc1, yc1, z2v, xc2, yc2)
                        else:
                            if supp_x2 == True and supp_z2 == True: # orthogonal lines
                                integ = a1 * b1 * a2 * c2 * integ_line_line_ortho_xy(z1v, yc1, xc1, zc2, y2v, xc2)
                            else:
                                integ = a1 * b1 * b2 * c2 * integ_line_line_ortho_xy(z1v, xc1, yc1, zc2, x2v, yc2)
    elif sum_supp == 3:  # point-volume or surface-line
        is_point_v1 = False
        if (supp_x1 + supp_y1 + supp_z1 == 3):
            is_point_v1 = True
        is_point_v2 = False
        if (supp_x2 + supp_y2 + supp_z2 == 3):
            is_point_v2 = True
        is_surf_v1 = False
        if (supp_x1 + supp_y1 + supp_z1 == 1):
            is_surf_v1 = True
        if is_point_v1 == True: # point - volume case
            integ = a1 * b1 * c1 * integ_point_vol(xc1, yc1, zc1, x2v, y2v, z2v)
        else:
            if is_point_v2 == True: # point - volume case
                integ = a2 * b2 * c2 * integ_point_vol(xc2, yc2, zc2, x1v, y1v, z1v)
            else:  # line-surface case
                if is_surf_v1 == True:  # bar1 is a surface
                    if supp_x1 == True:  # bar1 is a surface in y-z plane
                        if supp_x2 == False:  # bar 2 is a line along x
                            integ = a1 * b2 * c2 * integ_line_surf_ortho(x2v, yc2, zc2, xc1, y1v, z1v)
                        else:
                            if supp_y2 == False:  # bar 2 is a line along y
                                integ = a1 * a2 * c2 * integ_line_surf_para(y1v, z1v, xc1, y2v, zc2, xc2)
                            else:  # bar 2 is a line along z
                                integ = a1 * a2 * b2 * integ_line_surf_para(z1v, y1v, xc1, z2v, yc2, xc2)
                    else:
                        if supp_y1 == True:  # bar1 is a surface in x-z plane
                            if supp_x2 == False:  # bar 2 is a line along x
                                integ = b1 * b2 * c2 * integ_line_surf_para(x1v, z1v, yc1, x2v, zc2, yc2)
                            else:
                                if supp_y2 == False:  # bar 2 is a line along y
                                    integ = b1 * a2 * c2 * integ_line_surf_ortho(y2v, xc2, zc2, yc1, x1v, z1v)
                                else:  # bar 2 is a line along z
                                    integ = b1 * a2 * b2 * integ_line_surf_para(z1v, x1v, yc1, z2v, xc2, yc2)
                        else: # bar1 is a surface in x-y plane
                            if supp_x2 == False:  # bar 2 is a line along x
                                integ = c1 * b2 * c2 * integ_line_surf_para(x1v, y1v, zc1, x2v, yc2, zc2)
                            else:
                                if supp_y2 == False:  # bar 2 is a line along y
                                    integ = c1 * a2 * c2 * integ_line_surf_para(y1v, x1v, zc1, y2v, xc2, zc2)
                                else:  # bar 2 is a line along z
                                    integ = c1 * a2 * b2 * integ_line_surf_ortho(z2v, xc2, yc2, zc1, x1v, y1v)
                else:  # bar2 is a surface
                    if supp_x2 == True:  # bar2 is a surface in y-z plane
                        if supp_x1 == False:  # bar 1 is a line along x
                            integ = a2 * b1 * c1 * integ_line_surf_ortho(x1v, yc1, zc1, xc2, y2v, z2v)
                        else:
                            if supp_y1 == False:  # bar 1 is a line along y
                                integ = a2 * a1 * c1 * integ_line_surf_para(y2v, z2v, xc2, y1v, zc1, xc1)
                            else:  # bar 1 is a line along z
                                integ = a2 * a1 * b1 * integ_line_surf_para(z2v, y2v, xc2, z1v, yc1, xc1)
                    else:
                        if supp_y2 == True:  # bar2 is a surface in x-z plane
                            if supp_x1 == False:  # bar 1 is a line along x
                                integ = b2 * b1 * c1 * integ_line_surf_para(x2v, z2v, yc2, x1v, zc1, yc1)
                            else:
                                if supp_y1 == False:  # bar 1 is a line along y
                                    integ = b2 * a1 * c1 * integ_line_surf_ortho(y1v, xc1, zc1, yc2, x2v, z2v)
                                else:  # bar 1 is a line along z
                                    integ = b2 * a1 * b1 * integ_line_surf_para(z2v, x2v, yc2, z1v, xc1, yc1)
                        else:  # bar2 is a surface in x-y plane
                            if supp_x1 == False:  # bar 1 is a line along x
                                integ = c2 * b1 * c1 * integ_line_surf_para(x2v, y2v, zc2, x1v, yc1, zc1)
                            else:
                                if supp_y1 == False:  # bar 1 is a line along y
                                    integ = c2 * a1 * c1 * integ_line_surf_para(y2v, x2v, zc2, y1v, xc1, zc1)
                                else:  # bar 1 is a line along z
                                    integ = c2 * a1 * b1 * integ_line_surf_ortho(z1v, xc1, yc1, zc2, x2v, y2v)
    elif sum_supp == 2:  # line-volume or surface-surface
        is_line_v1 = False
        if (supp_x1 + supp_y1 + supp_z1 == 2):
            is_line_v1 = True
        is_line_v2 = False
        if (supp_x2 + supp_y2 + supp_z2 == 2):
            is_line_v2 = True
        if is_line_v1 == True:  # bar1 is a line
            if supp_x1 == False:  # bar1 is a line along x
                integ = b1 * c1 * integ_line_vol(x2v, y2v, z2v, x1v, yc1, zc1)
            else:
                if supp_y1 == False:  # bar1 is a line along y
                    integ = a1 * c1 * integ_line_vol(y2v, x2v, z2v, y1v, xc1, zc1)
                else:  # bar1 is a line along z
                    integ = a1 * b1 * integ_line_vol(z2v, x2v, y2v, z1v, xc1, yc1)
        else:
            if is_line_v2 == True:  # bar2 is a line
                if supp_x2 == False:  # bar2 is a line along x
                    integ = b2 * c2 * integ_line_vol(x1v, y1v, z1v, x2v, yc2, zc2)
                else:
                    if supp_y2 == False:  # bar2 is a line along y
                        integ = a2 * c2 * integ_line_vol(y1v, x1v, z1v, y2v, xc2, zc2)
                    else:  # bar2 is a line along z
                        integ = a2 * b2 * integ_line_vol(z1v, x1v, y1v, z2v, xc2, yc2)
            else:  # surface-surface case
                if supp_x1 == True:  # bar1 is a surface in yz plane
                    if supp_x2 == True:  # bar2 is a surface in yz plane
                        integ = a1 * a2 * integ_surf_surf_para(y1v, z1v, xc1, y2v, z2v, xc2)
                    else:
                        if supp_y2 == True:  # bar2 is a surface in xz plane
                            integ = a1 * b2 * integ_surf_surf_ortho(z1v, y1v, xc1, z2v, yc2, x2v)
                        else:  # bar2 is a surface in xy plane
                            integ = a1 * c2 * integ_surf_surf_ortho(y1v, z1v, xc1, y2v, zc2, x2v)
                else:
                    if supp_y1 == True:  # bar1 is a surface in xz plane
                        if supp_x2 == True:  # bar2 is a surface in yz plane
                            integ = b1 * a2 * integ_surf_surf_ortho(z1v, x1v, yc1, z2v, xc2, y2v)
                        else:
                            if supp_y2 == True:  # bar2 is a surface in xz plane
                                integ = b1 * b2 * integ_surf_surf_para(x1v, z1v, yc1, x2v, z2v, yc2)
                            else:  # bar2 is a surface in xy plane
                                integ = b1 * c2 * integ_surf_surf_ortho(x1v, z1v, yc1, x2v, zc2, y2v)
                    else:  # bar1 is a surface in xy plane

                        if supp_x2 == True:  # bar2 is a surface in yz plane
                            integ = c1 * a2 * integ_surf_surf_ortho(y1v, x1v, zc1, y2v, xc2, z2v)
                        else:
                            if supp_y2 == True:  # bar2 is a surface in xz plane
                                integ = c1 * b2 * integ_surf_surf_ortho(x1v, y1v, zc1, x2v, yc2, z2v)
                            else:  # bar2 is a surface in xy plane
                                integ = c1 * c2 * integ_surf_surf_para(x1v, y1v, zc1, x2v, y2v, zc2)
    elif sum_supp == 1:  # surface-volume case
        if supp_x1 == True:  # bar1 is a surface in yz plane
            integ = a1 * integ_vol_surf(y2v, z2v, x2v, y1v, z1v, xc1)
        else:
            if supp_y1 == True:  # bar1 is a surface in xz plane
                integ = b1 * integ_vol_surf(x2v, z2v, y2v, x1v, z1v, yc1)
            else:
                if supp_z1 == True:  # bar1 is a surface in xy plane
                    integ = c1 * integ_vol_surf(x2v, y2v, z2v, x1v, y1v, zc1)
                else:
                    if supp_x2 == True:  # bar2 is a surface in yz plane
                        integ = a2 * integ_vol_surf(y1v, z1v, x1v, y2v, z2v, xc2)
                    else:
                        if supp_y2 == True:  # bar2 is a surface in xz plane
                            integ = b2 * integ_vol_surf(x1v, z1v, y1v, x2v, z2v, yc2)
                        else: # bar2 is a surface in xy plane
                            integ = c2 * integ_vol_surf(x1v, y1v, z1v, x2v, y2v, zc2)
    else:  # volume - volume case
        integ = integ_vol_vol(x1v, y1v, z1v, x2v, y2v, z2v)

    return integ

#@jit(nopython=True, cache=True, parallel=True, fastmath=True)
@jit(nopython=True, cache=True, fastmath=True)
def compute_Lp_matrix(bars,sizex,sizey,sizez,dc):

    N = bars.shape[0]
    mat_Lp=np.zeros((N, N), dtype='double')

    S = sizey * sizez * sizey * sizez
    if dc == 2:
        S = sizex * sizez * sizex * sizez
    else:
        if dc == 3:
            S = sizex * sizey * sizex * sizey

    for m in range(N):
        for n in range(m, N):
            if m==n:
                if dc == 1:
                    l = abs(bars[m, 3] - bars[m, 0])
                    w = abs(bars[m, 4] - bars[m, 1])
                    t = abs(bars[m, 5] - bars[m, 2])
                elif dc == 2:
                    w = abs(bars[m, 3] - bars[m, 0])
                    l = abs(bars[m, 4] - bars[m, 1])
                    t = abs(bars[m, 5] - bars[m, 2])
                elif dc == 3:
                    t = abs(bars[m, 3] - bars[m, 0])
                    w = abs(bars[m, 4] - bars[m, 1])
                    l = abs(bars[m, 5] - bars[m, 2])

                mat_Lp[m, n] = compute_Lp_self(l, w, t)
            else:
                mat_Lp[m, n] = 1e-7 / S * Integ_vol_vol([bars[m,0],bars[m,3]], [bars[m,1],bars[m,4]], [bars[m,2],bars[m,5]],
                                                    [bars[n,0],bars[n,3]], [bars[n,1],bars[n,4]], [bars[n,2],bars[n,5]])
                mat_Lp[n, m] = mat_Lp[m, n]

    return mat_Lp