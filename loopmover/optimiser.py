import copy
import random
import sys
import tempfile

import ampal
import matplotlib.pyplot as plt
import numpy
import openmm as mm
from ampal.geometry import Quaternion, angle_between_vectors, dihedral
from openmm import app, unit, vec3

from loopmover.specification import CyclicPeptide


def toroidal_dist(point1, point2):

    xdiff = abs(point1[0] - point2[0])
    if xdiff > 180:
        xdiff = 360 - xdiff

    ydiff = abs(point1[1] - point2[1])
    if ydiff > 180:
        ydiff = 360 - ydiff

#     return xdiff, ydiff
    return(numpy.sqrt(xdiff**2 + ydiff**2))


def calc_rmsd2(rama1, rama2):
    dist = [toroidal_dist(x, y) for x, y in zip(rama1, rama2)]
    
    squares = [x**2 for x in dist]
    
    rmsd = numpy.sqrt(sum(squares)/len(dist))

    return(rmsd)


# this is the crazy code for loop moving (kinematic closure)
# paper at https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.10416
# there is some code at the start/end to let it interface with openmm, but all the mad matrix stuff is original
# makes use of a load of sub-functions further down

def find_alternate_positions(mod, positions, index1, index2, index3):
# returns a list of new positions that can be applied to the model
    """
    Needs tidied
    """
    start_pos = copy.deepcopy(positions)
    residues = [r for r in mod.topology.residues()]
    atoms1 = list(residues[index1].atoms())
    atoms2 = list(residues[index2].atoms())
    atoms3 = list(residues[index3].atoms())
    Natom_ind = [a.index for a in atoms1+atoms2+atoms3 if a.name == 'N']
    CAatom_ind = [a.index for a in atoms1+atoms2+atoms3 if a.name == 'CA']
    Catom_ind = [a.index for a in atoms1+atoms2+atoms3 if a.name == 'C']
    #the above works but seems iffy, why not just index each set of atoms separately?
   
    Natom_vecs = [numpy.array([v._value for v in positions[x]]) for x in Natom_ind]
    CAatom_vecs = [numpy.array([v._value for v in positions[x]]) for x in CAatom_ind]
    Catom_vecs = [numpy.array([v._value for v in positions[x]]) for x in Catom_ind]
    
    residues = [r for r in mod.topology.residues()]
    initlist = [x for x in range(len(residues))]
    if index1 > index3:
        l13 = initlist[initlist.index(index1):] + initlist[:initlist.index(index3)+1]
    else:
        l13 = initlist[index1:index3+1]
    l12 = l13[:l13.index(index2)+1]
    l23 = l13[l13.index(index2):]

    
    indexlist1 = []
#     indexlist1 += [at.index for at in residues[l12[0]].atoms() if at.name not in ['N', 'H', 'CA']]
    indexlist1 += [at.index for at in residues[l12[0]].atoms() if at.name in ['C', 'O']]
# central 
#     indexlist1 += [at.index for at in residues[l12[-1]].atoms() if at.name not in ['C', 'O', 'CA']]
    indexlist1 += [at.index for at in residues[l12[-1]].atoms() if at.name in ['N', 'H']]

    for i in l12[1:-1]:
        indexlist1+=[at.index for at in residues[i].atoms()]
    sc_1 = [at.index for at in residues[index1].atoms() if at.name not in ['C', 'CA', 'O', 'N', 'H']]

    indexlist2 = []
# central
#     indexlist2 += [at.index for at in residues[l23[0]].atoms() if at.name not in ['N', 'H', 'CA']]
    indexlist2 += [at.index for at in residues[l23[0]].atoms() if at.name in ['C', 'O']]

#     indexlist2 += [at.index for at in residues[l23[-1]].atoms() if at.name not in ['C', 'O', 'CA']]
    indexlist2 += [at.index for at in residues[l23[-1]].atoms() if at.name in ['N', 'H']]
    
    for i in l23[1:-1]:
        indexlist2+=[at.index for at in residues[i].atoms()]
    sc_2 = [at.index for at in residues[index2].atoms() if at.name not in ['C', 'CA', 'O', 'N', 'H']]


    indexlist3 = []
#     indexlist3 += [at.index for at in residues[l13[0]].atoms() if at.name not in ['N', 'H', 'CA']]
    indexlist3 += [at.index for at in residues[l13[0]].atoms() if at.name in ['C', 'O']]

    indexlist3 += [at.index for at in residues[l13[-1]].atoms() if at.name in ['N', 'H']]
#     indexlist3 += [at.index for at in residues[l13[-1]].atoms() if at.name not in ['C', 'O', 'CA']]

    for i in l13[1:-1]:
        indexlist3+=[at.index for at in residues[i].atoms()]
    sc_3 = [at.index for at in residues[index3].atoms() if at.name not in ['C', 'CA', 'O', 'N', 'H']]

        

    indexlist = [indexlist1, indexlist2, indexlist3]

    # gives [2, 0, 1] and [1, 2, 0] respectively (could have just written those I suppose)
    minus_idx = [x % 3 for x in range(-1, 2)]
    plus_idx = [x % 3 for x in range(1, 4)]

    # alternative definition of alpha etc direct from atoms using angle between vectors
    alphas = numpy.array([angle_between_vectors((CAatom_vecs[y] - CAatom_vecs[x]), (CAatom_vecs[x] - CAatom_vecs[z]), radians=True)
                          for x, y, z in zip(range(3), plus_idx, minus_idx)])
    alphas[1] = numpy.pi - alphas[1]
    etas = numpy.array([angle_between_vectors((Catom_vecs[x] - CAatom_vecs[x]), (CAatom_vecs[y] - CAatom_vecs[x]), radians=True)
                        for x, y in zip(range(3), plus_idx)])
    etas[2] = numpy.pi - etas[2]


    xis = numpy.array([angle_between_vectors((CAatom_vecs[y] - CAatom_vecs[x]), (Natom_vecs[x] - CAatom_vecs[x]), radians=True)
                       for x, y in zip(range(3), minus_idx)])
    xis[0] = numpy.pi - xis[0]
    deltas = numpy.array([dihedral(Catom_vecs[x], CAatom_vecs[x], CAatom_vecs[y], Natom_vecs[y], radians=True)
                          for x, y in zip(minus_idx, range(3))])
    # deltas is indexed differently here (in contrast to the paper definitions) as original code seems to need it thus
    deltas[1] = numpy.pi - deltas[1]
    deltas[2] = numpy.pi - deltas[2]

    zhats = numpy.array([(CAatom_vecs[y] - CAatom_vecs[x]) /
                         numpy.sqrt(sum((CAatom_vecs[y] - CAatom_vecs[x]) ** 2)) for x, y in zip(range(3), plus_idx)])
    yhat = numpy.cross(zhats[2], zhats[0]) / numpy.sqrt(sum(numpy.cross(zhats[2], zhats[0]) ** 2))

    ref_tau1 = numpy.pi - dihedral(Catom_vecs[2], CAatom_vecs[0], CAatom_vecs[2], (CAatom_vecs[2] + yhat), radians=True)
    ref_tau2 = dihedral((CAatom_vecs[0] - yhat), CAatom_vecs[0], CAatom_vecs[1], Catom_vecs[0], radians=True)
    ref_tau3 = dihedral((CAatom_vecs[1] + yhat), CAatom_vecs[1], CAatom_vecs[2], Catom_vecs[1], radians=True)
    ref_taus = [ref_tau1, ref_tau2, ref_tau3]

    thetas = numpy.array(
        [angle_between_vectors((CAatom_vecs[x] - Natom_vecs[x]), (CAatom_vecs[x] - Catom_vecs[x]), radians=True) for x in range(3)])
    
    B0 = numpy.zeros(3)
    B1 = numpy.zeros(3)
    B2 = numpy.zeros(3)
    B3 = numpy.zeros(3)
    B4 = numpy.zeros(3)
    B5 = numpy.zeros(3)
    B6 = numpy.zeros(3)
    B7 = numpy.zeros(3)
    B8 = numpy.zeros(3)

    C0 = numpy.zeros([3, 3])
    C1 = numpy.zeros([3, 3])
    C2 = numpy.zeros([3, 3])

    u11 = numpy.zeros([5, 5])
    u12 = numpy.zeros([5, 5])
    u13 = numpy.zeros([5, 5])
    u31 = numpy.zeros([5, 5])
    u32 = numpy.zeros([5, 5])
    u33 = numpy.zeros([5, 5])

    for i in range(3):
        A0 = numpy.cos(alphas[i]) * numpy.cos(xis[i]) * numpy.cos(etas[i]) - numpy.cos(thetas[i])
        A1 = -numpy.sin(alphas[i]) * numpy.cos(xis[i]) * numpy.sin(etas[i])
        A2 = numpy.sin(alphas[i]) * numpy.sin(xis[i]) * numpy.cos(etas[i])
        A3 = numpy.sin(xis[i]) * numpy.sin(etas[i])
        A4 = A3 * numpy.cos(alphas[i])

        A21 = A2 * numpy.cos(deltas[i])
        A22 = A2 * numpy.sin(deltas[i])
        A31 = A3 * numpy.cos(deltas[i])
        A32 = A3 * numpy.sin(deltas[i])
        A41 = A4 * numpy.cos(deltas[i])
        A42 = A4 * numpy.sin(deltas[i])

        B0[i] = A0 + A22 + A31
        B1[i] = 2 * (A1 + A42)
        B2[i] = 2 * (A32 - A21)
        B3[i] = -4 * A41
        B4[i] = A0 + A22 - A31
        B5[i] = A0 - A22 - A31
        B6[i] = -2 * (A21 + A32)
        B7[i] = 2 * (A1 - A42)
        B8[i] = A0 - A22 + A31

    C0[0][0] = B0[0]
    C0[0][1] = B2[0]
    C0[0][2] = B5[0]
    C1[0][0] = B1[0]
    C1[0][1] = B3[0]
    C1[0][2] = B7[0]
    C2[0][0] = B4[0]
    C2[0][1] = B6[0]
    C2[0][2] = B8[0]

    for i in range(1, 3):
        C0[i][0] = B0[i]
        C0[i][1] = B1[i]
        C0[i][2] = B4[i]
        C1[i][0] = B2[i]
        C1[i][1] = B3[i]
        C1[i][2] = B6[i]
        C2[i][0] = B5[i]
        C2[i][1] = B7[i]
        C2[i][2] = B8[i]

    for i in range(3):
        u11[0][i] = C0[0][i]
        u12[0][i] = C1[0][i]
        u13[0][i] = C2[0][i]
        u31[i][0] = C0[1][i]
        u32[i][0] = C1[1][i]
        u33[i][0] = C2[1][i]

    p1 = [2, 0]
    p3 = [0, 2]

    um1, p_um1 = poly_mul_sub2(u32, u32, u31, u33, p3, p3, p3, p3)
    um2, p_um2 = poly_mul_sub2(u12, u32, u11, u33, p1, p3, p1, p3)
    um3, p_um3 = poly_mul_sub2(u12, u33, u13, u32, p1, p3, p1, p3)
    um4, p_um4 = poly_mul_sub2(u11, u33, u31, u13, p1, p3, p3, p1)
    um5, p_um5 = poly_mul_sub2(u13, um1, u33, um2, p1, p_um1, p3, p_um2)
    um6, p_um6 = poly_mul_sub2(u13, um4, u12, um3, p1, p_um4, p1, p_um3)
    q_tmp, _p_Q = poly_mul_sub2(u11, um5, u31, um6, p1, p_um5, p3, p_um6)

    Q = q_tmp[:]

    R = numpy.zeros([3, 17])

    for i in range(3):
        R[0][i] = C0[2][i]
        R[1][i] = C1[2][i]
        R[2][i] = C2[2][i]
    p2 = 2
    p4 = 4

    f1, p_f1 = poly_mul_sub1(R[1], R[1], R[0], R[2], p2, p2, p2, p2)
    f2, p_f2 = poly_mul1(R[1], R[2], p2, p2)
    f3, p_f3 = poly_mul_sub1(R[1], f1, R[0], f2, p2, p_f1, p2, p_f2)
    f4, p_f4 = poly_mul1(R[2], f1, p2, p_f1)
    f5, p_f5 = poly_mul_sub1(R[1], f3, R[0], f4, p2, p_f3, p2, p_f4)

    f6, p_f6 = poly_mul_sub1(Q[1], R[1], Q[0], R[2], p4, p2, p4, p2)
    f7, p_f7 = poly_mul_sub1(Q[2], f1, R[2], f6, p4, p_f1, p2, p_f6)
    f8, p_f8 = poly_mul_sub1(Q[3], f3, R[2], f7, p4, p_f3, p2, p_f7)
    f9, p_f9 = poly_mul_sub1(Q[4], f5, R[2], f8, p4, p_f5, p2, p_f8)

    f10, p_f10 = poly_mul_sub1(Q[3], R[1], Q[4], R[0], p4, p2, p4, p2)
    f11, p_f11 = poly_mul_sub1(Q[2], f1, R[0], f10, p4, p_f1, p2, p_f10)
    f12, p_f12 = poly_mul_sub1(Q[1], f3, R[0], f11, p4, p_f3, p2, p_f11)

    f13, p_f13 = poly_mul_sub1(Q[2], R[1], Q[1], R[2], p4, p2, p4, p2)
    f14, p_f14 = poly_mul_sub1(Q[3], f1, R[2], f13, p4, p_f1, p2, p_f13)
    f15, p_f15 = poly_mul_sub1(Q[3], R[1], Q[2], R[2], p4, p2, p4, p2)
    f16, p_f16 = poly_mul_sub1(Q[4], f1, R[2], f15, p4, p_f1, p2, p_f15)
    f17, p_f17 = poly_mul_sub1(Q[1], f14, Q[0], f16, p4, p_f14, p4, p_f16)

    f18, p_f18 = poly_mul_sub1(Q[2], R[2], Q[3], R[1], p4, p2, p4, p2)
    f19, p_f19 = poly_mul_sub1(Q[1], R[2], Q[3], R[0], p4, p2, p4, p2)
    f20, p_f20 = poly_mul_sub1(Q[3], f19, Q[2], f18, p4, p_f19, p4, p_f18)
    f21, p_f21 = poly_mul_sub1(Q[1], R[1], Q[2], R[0], p4, p2, p4, p2)
    f22, p_f22 = poly_mul1(Q[4], f21, p4, p_f21)
    f23, p_f23 = poly_sub1(f20, f22, p_f20, p_f22)
    f24, p_f24 = poly_mul1(R[0], f23, p2, p_f23)
    f25, p_f25 = poly_sub1(f17, f24, p_f17, p_f24)
    f26, p_f26 = poly_mul_sub1(Q[4], f12, R[2], f25, p4, p_f12, p2, p_f25)
    poly_coeff, p_final = poly_mul_sub1(Q[0], f9, R[0], f26, p4, p_f9, p2, p_f26)

    if p_final != 16:
        raise ValueError('Degree of polynomial is not 16')

    if poly_coeff[16] < 0.0:
        for i in range(17):
            poly_coeff[i] *= -1

    poly_coeff2 = list(poly_coeff)
    poly_coeff2.reverse()
    all_roots = numpy.roots(poly_coeff2)
    roots = all_roots[numpy.isreal(all_roots)].real
    net_rotation_list = []
    for i in range(len(roots)):
        half_tan = numpy.zeros(3)
        half_tan[2] = roots[i]
        half_tan[1] = calc_t2(half_tan[2], Q, R)
        half_tan[0] = calc_t1(half_tan[2], half_tan[1], C0, C1, C2)

        cos_tau = numpy.zeros(4)
        sin_tau = numpy.zeros(4)
        for j in range(1, 4):
            ht = half_tan[j - 1]
            tmp = 1.0 + ht ** 2
            cos_tau[j] = (1.0 - ht ** 2) / tmp
            sin_tau[j] = 2.0 * ht / tmp
        cos_tau[0] = cos_tau[3]
        sin_tau[0] = sin_tau[3]
        taus = [numpy.arctan2(sin_tau[i], cos_tau[i]) for i in range(3)]
        net_rotations = [ref_taus[0] - taus[0]]
        net_rotations += [taus[j] - ref_taus[j] for j in range(1, 3)]
        net_rotation_list.append([net_rotations[1], net_rotations[2], net_rotations[0]])
        #HOW CAN THIS MESS UP CB AND HB BUT NOT O OR H???
    ax_ind = [(CAatom_ind[1], CAatom_ind[0]), (CAatom_ind[2], CAatom_ind[1]), (CAatom_ind[2], CAatom_ind[0])] 

    axes = []
    rotation_points = []
    for i in range(len(ax_ind)):
        a1 = numpy.array([v._value for v in start_pos[ax_ind[i][0]]])
        a2 = numpy.array([v._value for v in start_pos[ax_ind[i][1]]])

        axis = a1 - a2
        axes.append(axis)
        rotation_points.append(a2)

    output_list = []
    for i in range(len(net_rotation_list)):
        output_pos = copy.deepcopy(start_pos)
        for j in range(3):
#         for j in [2, 0, 1]:
            
#             # builds the new axis for rotation off latest CA positions

            quat = Quaternion.angle_and_axis(angle=net_rotation_list[i][j], axis=axes[j], radians=True)
            for k in indexlist[j]:
                start_vec = numpy.array([v._value for v in output_pos[k]])
                rotated_vec = quat.rotate_vector(v=start_vec, point=rotation_points[j])
                newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
                output_pos[k] = newpos
#         output_list.append(output_pos)
    # now move side chains of index residues
    # this can work but has to take two steps...
    # align dot product of 
    #penguin
#         Natom_ind = [a.index for a in atoms1+atoms2+atoms3 if a.name == 'N']
#         CAatom_ind = [a.index for a in atoms1+atoms2+atoms3 if a.name == 'CA']

        n_1_old_v = numpy.array([v._value for v in start_pos[Natom_ind[0]]])
        ca_1_old_v = numpy.array([v._value for v in start_pos[CAatom_ind[0]]])
        c_1_old_v = numpy.array([v._value for v in start_pos[Catom_ind[0]]])
        old_n_1_c_1_v = n_1_old_v - c_1_old_v
        old_1_v = ca_1_old_v - (n_1_old_v + c_1_old_v)/2
        old_1_cross = numpy.cross(old_1_v, old_n_1_c_1_v)
        old_1_x_ca = ca_1_old_v - old_1_cross 

        n_2_old_v = numpy.array([v._value for v in start_pos[Natom_ind[1]]])
        ca_2_old_v = numpy.array([v._value for v in start_pos[CAatom_ind[1]]])
        c_2_old_v = numpy.array([v._value for v in start_pos[Catom_ind[1]]])
        old_n_2_c_2_v = n_2_old_v - c_2_old_v
        old_2_v = ca_2_old_v - (n_2_old_v + c_2_old_v)/2
        old_2_cross = numpy.cross(old_2_v, old_n_2_c_2_v)
        old_2_x_ca = ca_2_old_v - old_2_cross 

        n_3_old_v = numpy.array([v._value for v in start_pos[Natom_ind[2]]])
        ca_3_old_v = numpy.array([v._value for v in start_pos[CAatom_ind[2]]])
        c_3_old_v = numpy.array([v._value for v in start_pos[Catom_ind[2]]])
        old_n_3_c_3_v = n_3_old_v - c_3_old_v
        old_3_v = ca_3_old_v - (n_3_old_v + c_3_old_v)/2
        old_3_cross = numpy.cross(old_3_v, old_n_3_c_3_v)
        old_3_x_ca = ca_3_old_v - old_3_cross 

        n_1_new_v = numpy.array([v._value for v in output_pos[Natom_ind[0]]])
        ca_1_new_v = numpy.array([v._value for v in output_pos[CAatom_ind[0]]])
        c_1_new_v = numpy.array([v._value for v in output_pos[Catom_ind[0]]])
        new_n_1_c_1_v = n_1_new_v - c_1_new_v
        new_1_v = ca_1_new_v - (n_1_new_v + c_1_new_v)/2
        new_1_cross = numpy.cross(new_1_v, new_n_1_c_1_v)
        new_1_x_ca = ca_1_new_v - new_1_cross 

        n_2_new_v = numpy.array([v._value for v in output_pos[Natom_ind[1]]])
        ca_2_new_v = numpy.array([v._value for v in output_pos[CAatom_ind[1]]])    
        c_2_new_v = numpy.array([v._value for v in output_pos[Catom_ind[1]]])
        new_n_2_c_2_v = n_2_new_v - c_2_new_v
        new_2_v = ca_2_new_v - (n_2_new_v + c_2_new_v)/2
        new_2_cross = numpy.cross(new_2_v, new_n_2_c_2_v)
        new_2_x_ca = ca_2_new_v - new_2_cross 

        n_3_new_v = numpy.array([v._value for v in output_pos[Natom_ind[2]]])
        ca_3_new_v = numpy.array([v._value for v in output_pos[CAatom_ind[2]]])    
        c_3_new_v = numpy.array([v._value for v in output_pos[Catom_ind[2]]])
        new_n_3_c_3_v = n_3_new_v - c_3_new_v
        new_3_v = ca_3_new_v - (n_3_new_v + c_3_new_v)/2
        new_3_cross = numpy.cross(new_3_v, new_n_3_c_3_v)
        new_3_x_ca = ca_3_new_v - new_3_cross 
      
        t1 = ampal.geometry.find_transformations(old_1_x_ca, ca_1_old_v, new_1_x_ca, ca_1_new_v)
        t2 = ampal.geometry.find_transformations(old_2_x_ca, ca_2_old_v, new_2_x_ca, ca_2_new_v)
        t3 = ampal.geometry.find_transformations(old_3_x_ca, ca_3_old_v, new_3_x_ca, ca_3_new_v)

        sc1_pos = [start_pos[_]._value for _ in sc_1]
        sc2_pos = [start_pos[_]._value for _ in sc_2]
        sc3_pos = [start_pos[_]._value for _ in sc_3]

        q1 = Quaternion.angle_and_axis(angle=t1[1], axis=t1[2])
        for j in range(len(sc_1)):
            rotated_vec = q1.rotate_vector(v=sc1_pos[j], point=t1[3])
            rotated_vec += t1[0]
            newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
            output_pos[sc_1[j]] = newpos
        
        q2 = Quaternion.angle_and_axis(angle=t2[1], axis=t2[2])
        for j in range(len(sc_2)):
            rotated_vec = q2.rotate_vector(v=sc2_pos[j], point=t2[3])
            rotated_vec += t2[0]
            newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
            output_pos[sc_2[j]] = newpos
        
        q3 = Quaternion.angle_and_axis(angle=t3[1], axis=t3[2])
        for j in range(len(sc_3)):
            rotated_vec = q3.rotate_vector(v=sc3_pos[j], point=t3[3])
            rotated_vec += t3[0]
            newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
            output_pos[sc_3[j]] = newpos
            
        # now have side chains in right position, but need 'stood up' to get correct tetrahedral geom
        cb_atom_ind = [a.index for a in atoms1+atoms2+atoms3 if a.name in ('CB', 'HA3')]
        ha_atom_ind = [a.index for a in atoms1+atoms2+atoms3 if a.name in ('HA', 'HA2')]
        
        cb_1_new_v = numpy.array([v._value for v in output_pos[cb_atom_ind[0]]])
        ha_1_new_v = numpy.array([v._value for v in output_pos[ha_atom_ind[0]]])

        cb_2_new_v = numpy.array([v._value for v in output_pos[cb_atom_ind[1]]])
        ha_2_new_v = numpy.array([v._value for v in output_pos[ha_atom_ind[1]]])
        
        cb_3_new_v = numpy.array([v._value for v in output_pos[cb_atom_ind[2]]])
        ha_3_new_v = numpy.array([v._value for v in output_pos[ha_atom_ind[2]]])
        
        ca_1_cb_1 = ampal.geometry.unit_vector(cb_1_new_v - ca_1_new_v)
        ca_1_ha_1 = ampal.geometry.unit_vector(ha_1_new_v - ca_1_new_v)
        
        ca_2_cb_2 = ampal.geometry.unit_vector(cb_2_new_v - ca_2_new_v)
        ca_2_ha_2 = ampal.geometry.unit_vector(ha_2_new_v - ca_2_new_v)
        
        ca_3_cb_3 = ampal.geometry.unit_vector(cb_3_new_v - ca_3_new_v)
        ca_3_ha_3 = ampal.geometry.unit_vector(ha_3_new_v - ca_3_new_v)
        
#         ref_p_1 = ca_1_new_v + ca_1_cb_1 + ca_1_ha_1
#         ref_p_2 = ca_2_new_v + ca_2_cb_2 + ca_2_ha_2
#         ref_p_3 = ca_3_new_v + ca_3_cb_3 + ca_3_ha_3
        
        # first term should be equal to vector from ca to ref_p_1, 
        # doesn't tell us which way to rotate!
        a1 = ampal.geometry.angle_between_vectors((ca_1_cb_1 + ca_1_ha_1), new_1_v)
        a2 = ampal.geometry.angle_between_vectors((ca_2_cb_2 + ca_2_ha_2), new_2_v)
        a3 = ampal.geometry.angle_between_vectors((ca_3_cb_3 + ca_3_ha_3), new_3_v)
        
#         ref1a = ampal.geometry.angle_between_vectors((ca_1_new_v - ha_1_new_v), (ca_1_new_v - c_1_new_v))
#         ref1b = ampal.geometry.angle_between_vectors((ca_1_new_v - ha_1_new_v), (ca_1_new_v - n_1_new_v))

        ref1a = ampal.geometry.distance(cb_1_new_v, c_1_new_v)
        ref1b = ampal.geometry.distance(cb_1_new_v, n_1_new_v)
        
        if ref1a > ref1b:
            a1 *= -1
            
#         ref2a = ampal.geometry.angle_between_vectors((ca_2_new_v - ha_2_new_v), (ca_2_new_v - c_2_new_v))
#         ref2b = ampal.geometry.angle_between_vectors((ca_2_new_v - ha_2_new_v), (ca_2_new_v - n_2_new_v))
        
        ref2a = ampal.geometry.distance(cb_2_new_v, c_2_new_v)
        ref2b = ampal.geometry.distance(cb_2_new_v, n_2_new_v)        
        
        if ref2a > ref2b:
            a2 *= -1
            
        ref3a = ampal.geometry.distance(cb_3_new_v, c_3_new_v)
        ref3b = ampal.geometry.distance(cb_3_new_v, n_3_new_v)        
        
        if ref3a > ref3b:
            a3 *= -1   
        
        # a1 and a2 are identical- ??? result of final rotation? OR something buggy/weird is going on in my code
        
        sc1_pos2 = [output_pos[_]._value for _ in sc_1]
        sc2_pos2 = [output_pos[_]._value for _ in sc_2]
        sc3_pos2 = [output_pos[_]._value for _ in sc_3]
        
        q1b = Quaternion.angle_and_axis(angle=a1, axis=new_1_cross) #(ca_1_ha_1 - ca_1_cb_1))
        for j in range(len(sc_1)):
            rotated_vec = q1b.rotate_vector(v=sc1_pos2[j], point=ca_1_new_v)
            newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
            output_pos[sc_1[j]] = newpos
            
        q2b = Quaternion.angle_and_axis(angle=a2, axis=new_2_cross) #(ca_2_ha_2 - ca_2_cb_2))
        for j in range(len(sc_2)):
            rotated_vec = q2b.rotate_vector(v=sc2_pos2[j], point=ca_2_new_v)
            newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
            output_pos[sc_2[j]] = newpos
            
        q3b = Quaternion.angle_and_axis(angle=a3, axis= new_3_cross) #(ca_3_ha_3 - ca_3_cb_3))
        for j in range(len(sc_3)):
            rotated_vec = q3b.rotate_vector(v=sc3_pos2[j], point=ca_3_new_v)
            newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
            output_pos[sc_3[j]] = newpos

        output_list.append(output_pos)

    return output_list



def poly_mul2(u1, u2, p1_in, p2_in):
    u3 = numpy.zeros([5, 5])
    p3 = [x + y for x, y in zip(p1_in, p2_in)]

    p11 = p1_in[0]
    p12 = p1_in[1]
    p21 = p2_in[0]
    p22 = p2_in[1]

    for i1 in range(p12 + 1):
        for j1 in range(p11 + 1):
            u1ij = u1[i1, j1]
            for i2 in range(p22 + 1):
                i3 = i1 + i2
                for j2 in range(p21 + 1):
                    j3 = j1 + j2
                    u3[i3, j3] = u3[i3, j3] + u1ij * u2[i2, j2]
    return u3, p3


def poly_sub2(u1, u2, p1_in, p2_in):
    p11 = p1_in[0]
    p12 = p1_in[1]
    p21 = p2_in[0]
    p22 = p2_in[1]

    p31 = max(p11, p21)
    p32 = max(p12, p22)
    p3 = [p31, p32]

    u3 = numpy.zeros([5, 5])

    for i in range(p32 + 1):
        i1_ok = (i > p12)
        i2_ok = (i > p22)
        for j in range(p31 + 1):
            if i2_ok or (j > p21):
                u3[i][j] = u1[i][j]
            elif i1_ok or (j > p11):
                u3[i][j] = -u2[i][j]
            else:
                u3[i][j] = u1[i][j] - u2[i][j]
    return u3, p3


def poly_mul_sub2(u1, u2, u3, u4, p1_in, p2_in, p3_in, p4_in):
    d1, pd1 = poly_mul2(u1, u2, p1_in, p2_in)
    d2, pd2 = poly_mul2(u3, u4, p3_in, p4_in)
    return poly_sub2(d1, d2, pd1, pd2)


def poly_mul1(u1, u2, p1_in, p2_in):
    p3 = p1_in + p2_in
    u3 = numpy.zeros(17)
    for i1 in range(p1_in + 1):
        u1i = u1[i1]
        for i2 in range(p2_in + 1):
            i3 = i1 + i2
            u3[i3] = u3[i3] + u1i * u2[i2]
    return u3, p3


def poly_sub1(u1, u2, p1_in, p2_in):
    p3 = max(p1_in, p2_in)
    u3 = numpy.zeros(17)
    for i in range(p3 + 1):
        if i > p2_in:
            u3[i] = -u2[i]
        else:
            u3[i] = u1[i] - u2[i]
    return u3, p3


def quaternion(axis, ang):
    tan_w = numpy.tan(ang)
    tan_sqr = tan_w ** 2
    tan1 = tan_sqr + 1.0
    _cosine = (1.0 - tan_sqr) / tan1
    _sine = 2 * tan_w / tan1
    p = [_cosine, axis[0] * _sine, axis[1] * _sine, axis[2] * _sine]
    return p


def rotation_matrix(quat):
    # need to get to grips with inbuild rot mat and quaternion, and hopefully swap these out
    b = [x * 2 for x in quat]
    q = numpy.zeros([4, 4])
    for i in range(4):
        for j in range(i, 4):
            q[i][j] = b[i] * quat[j]
    q[0][0] -= 1
    u = numpy.zeros([3, 3])
    u[0][0] = q[0][0] + q[1][1]
    u[0][1] = q[1][2] - q[0][3]
    u[0][2] = q[1][3] + q[0][2]

    u[1][0] = q[1][2] + q[0][3]
    u[1][1] = q[0][0] + q[2][2]
    u[1][2] = q[2][3] - q[0][1]

    u[2][0] = q[1][3] - q[0][2]
    u[2][1] = q[2][3] + q[0][1]
    u[2][2] = q[0][0] + q[3][3]

    return u


def poly_mul_sub1(u1, u2, u3, u4, p1_in, p2_in, p3_in, p4_in):
    d1, pd1 = poly_mul1(u1, u2, p1_in, p2_in)
    d2, pd2 = poly_mul1(u3, u4, p3_in, p4_in)
    return poly_sub1(d1, d2, pd1, pd2)


def sign(a, b):
    if b >= 0:
        return abs(a)
    else:
        return -abs(a)


def calc_t1(t0, t2, C0, C1, C2):
    t0_2 = t0 ** 2
    t2_2 = t2 ** 2

    U11 = C0[0][0] + C0[0][1] * t0 + C0[0][2] * t0_2
    U12 = C1[0][0] + C1[0][1] * t0 + C1[0][2] * t0_2
    U13 = C2[0][0] + C2[0][1] * t0 + C2[0][2] * t0_2
    U31 = C0[1][0] + C0[1][1] * t2 + C0[1][2] * t2_2
    U32 = C1[1][0] + C1[1][1] * t2 + C1[1][2] * t2_2
    U33 = C2[1][0] + C2[1][1] * t2 + C2[1][2] * t2_2

    tmp_value = (U31 * U13 - U11 * U33) / (U12 * U33 - U13 * U32)
    return tmp_value


def calc_t2(t0, Q, R):
    t0_2 = t0 * t0
    t0_3 = t0_2 * t0
    t0_4 = t0_3 * t0

    A0 = Q[0][0] + Q[0][1] * t0 + Q[0][2] * t0_2 + Q[0][3] * t0_3 + Q[0][4] * t0_4
    A1 = Q[1][0] + Q[1][1] * t0 + Q[1][2] * t0_2 + Q[1][3] * t0_3 + Q[1][4] * t0_4
    A2 = Q[2][0] + Q[2][1] * t0 + Q[2][2] * t0_2 + Q[2][3] * t0_3 + Q[2][4] * t0_4
    A3 = Q[3][0] + Q[3][1] * t0 + Q[3][2] * t0_2 + Q[3][3] * t0_3 + Q[3][4] * t0_4
    A4 = Q[4][0] + Q[4][1] * t0 + Q[4][2] * t0_2 + Q[4][3] * t0_3 + Q[4][4] * t0_4

    B0 = R[0][0] + R[0][1] * t0 + R[0][2] * t0_2
    B1 = R[1][0] + R[1][1] * t0 + R[1][2] * t0_2
    B2 = R[2][0] + R[2][1] * t0 + R[2][2] * t0_2

    B2_2 = B2 * B2
    B2_3 = B2_2 * B2

    K0 = A2 * B2 - A4 * B0
    K1 = A3 * B2 - A4 * B1
    K2 = A1 * B2_2 - K1 * B0
    K3 = K0 * B2 - K1 * B1

    tmp_value = (K3 * B0 - A0 * B2_3) / (K2 * B2 - K3 * B1)

    return tmp_value



# this is where we actually define the class for the optimiser
# __init__ just defines us the rmsd cutoff in ramachandran space, and the sequence to optimise

class CyclicPeptideOptimiser:
    
    def __init__(self, seq, forcefield='amber14-all.xml', implicit_solvent='implicit/obc2.xml'): #, rmsd_cut=50):
    # initialise
        
        self.seq = seq
        self.forcefield = forcefield
        self.implicit_solvent = implicit_solvent
#         self.rmsd_cut = rmsd_cut
        

    def build_start_mac(self):
        # We use CyclicPeptide to generate a starting backbone macrocycle.
        self.start_mac = CyclicPeptide(self.seq, auto_build=True)

    def amber_setup(self, forcefield=None, implicit_solvent=None, pH=7.0):
        if forcefield is not None:
            self.forcefield = forcefield
        if implicit_solvent is not None:
            self.implicit_solvent = implicit_solvent
            
        self.good = 0
        self.bad = 0
        self.flip = 0
        self.cis_bad = 0
        
        import os

        from ampal.amino_acids import standard_amino_acids
        from pdbfixer import PDBFixer
        
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as f:
            f.write(self.start_mac.pdb.encode())
            f_name = f.name
            
        fixer = PDBFixer(filename=f_name)
        os.remove(f_name)
        
        mutations = [f"GLY-{i+1}-{standard_amino_acids[aa.upper()]}" for i, aa in enumerate(self.seq)]
        fixer.applyMutations(mutations, "A")
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        
        self.model = app.Modeller(fixer.topology, fixer.positions)

        residues = [r for r in self.model.topology.residues()]
        # assumes no OH at end. Would need to target and remove atoms for that if present
        self.model.topology.addBond(next(at for at in residues[0].atoms() if at.name == 'N'),
                                    next(at for at in residues[-1].atoms() if at.name == 'C'))
        self.model.addHydrogens(pH=pH)
        #need to re-generate residues to get fresh atoms
        excess_atoms = [a for r in self.model.topology.residues() for a in r.atoms() if a.name in ('H2', 'H3', 'OXT')]
        self.model.delete(excess_atoms)
        
        if self.implicit_solvent is not None:
            forcefield = app.ForceField(self.forcefield, self.implicit_solvent)
        else:
            forcefield = app.ForceField(self.forcefield)
        
        self.system = forcefield.createSystem(self.model.topology,
                                              nonbondedMethod=app.NoCutoff,
                                              constraints=None, ignoreExternalBonds=True)
                                              
        self.force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        self.force.addGlobalParameter("k", 1.0*unit.kilocalories_per_mole/unit.angstroms**2)
        self.force.addPerParticleParameter("x0")
        self.force.addPerParticleParameter("y0")
        self.force.addPerParticleParameter("z0")             

        atoms = [at for at in self.model.topology.atoms()]

        self.n_indices = [a.index for a in atoms if a.name == 'N']
        self.ca_indices = [a.index for a in atoms if a.name == 'CA']
        self.c_indices = [a.index for a in atoms if a.name == 'C']
        
        constrainedatoms = []
        for i, atom_crd in enumerate(self.model.positions):
            if atoms[i].name == 'CA':
                constrainedatoms.append((i, atoms[i]))
                self.force.addParticle(i, atom_crd.value_in_unit(unit.nanometers))
        self.system.addForce(self.force)

        residues2 = [r for r in self.model.topology.residues()]
        self.nonG_idx = [r.index for r in residues2 if r.name != 'GLY']
        self.n_ind = [next(a.index for a in r.atoms() if a.name == 'N') for r in residues2 if r.name != 'GLY']
        self.c_ind = [next(a.index for a in r.atoms() if a.name == 'C') for r in residues2 if r.name != 'GLY']
        self.ca_ind = [next(a.index for a in r.atoms() if a.name == 'CA') for r in residues2 if r.name != 'GLY']
        self.cb_ind = [next(a.index for a in r.atoms() if a.name == 'CB') for r in residues2 if r.name != 'GLY']
       
        integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
        integrator.setConstraintTolerance(0.00001)

        try:
            platform = mm.Platform.getPlatformByName('CUDA')
        except mm.OpenMMException:
            platform = mm.Platform.getPlatformByName('CPU')
        properties = {}
        
        self.simulation = app.Simulation(self.model.topology, self.system, integrator, platform, properties)

        self.simulation.context.setPositions(self.model.positions)
        startpos = copy.deepcopy(self.model.positions)
              
        if any(x in 'TtIi' for x in self.seq):
            startpos = self.sc_chir_check_flip(startpos)
            
        for k in range(len(self.seq)):
            chir = self.check_chirality(k, startpos)
            if (chir == 'L') and (str.islower(self.seq[k])) or (chir == 'D') and not (str.islower(self.seq[k])):
                startpos = self.invert_chirality(k, startpos)

        self.simulation.context.setPositions(startpos)
        for k in range(self.force.getNumParticles()):
            idx = self.force.getParticleParameters(k)[0]
            self.force.setParticleParameters(k, idx, startpos[idx].value_in_unit(unit.nanometers))
        self.force.updateParametersInContext(self.simulation.context)
        
        self.simulation.minimizeEnergy(maxIterations=1000)
        state = self.simulation.context.getState(getEnergy=True, getPositions=True)
        self.initial_state = (state.getPotentialEnergy() / unit.kilojoules_per_mole, state.getPositions())

    def flip_amide(self, positions, index, angle=180):

        """
        Flips an amide bond 180 degrees along the Ca-Ca axis.
        Will result in suspect geometry"""
        outpos = copy.deepcopy(positions)
        
        i1 = index
        i2 = (index+1)%len(self.seq)
        
        
#         start_pos = copy.deepcopy(positions)
        residues = [r for r in self.model.topology.residues()]
        
        ca1_i = next(a.index for a in residues[i1].atoms() if a.name == 'CA')
        ca2_i = next(a.index for a in residues[i2].atoms() if a.name == 'CA')
               
        ca1_v = positions[ca1_i]._value
        ca2_v = positions[ca2_i]._value
        
#         ca1_v = [a._value for a in residues[i1] if a.name == 'CA'][0]
#         ca2_v = [a._value for a in residues[i2] if a.name == 'CA'][0]
        
        c1c2axis = ca2_v - ca1_v
        
        c_i = next(a.index for a in residues[i1].atoms() if a.name == 'C')
        o_i = next(a.index for a in residues[i1].atoms() if a.name == 'O')
        n_i = next(a.index for a in residues[i2].atoms() if a.name == 'N')
        h_i = next(a.index for a in residues[i2].atoms() if a.name in ('H', 'CD'))
        
        idx_list = [c_i, o_i, n_i, h_i]
        
        c_v = positions[c_i]._value
        o_v = positions[o_i]._value
        n_v = positions[n_i]._value
        h_v = positions[h_i]._value
        
        rot_list = [c_v, o_v, n_v, h_v]
        
        quat = Quaternion.angle_and_axis(angle=angle, axis=c1c2axis)
        
        rotated = []
        for pos in rot_list:
            newpos = quat.rotate_vector(v=pos, point=ca1_v)
            newpos2 = unit.quantity.Quantity(vec3.Vec3(*[x for x in newpos]), unit=unit.nanometer)
            rotated.append(newpos2)
        
        for j in range(len(idx_list)):
            outpos[idx_list[j]] = rotated[j]
        
        return(outpos)
        
        
#         quat = Quaternion.angle_and_axis(angle=transforms[i][1], axis=transforms[i][2])
#         rotated_vec = quat.rotate_vector(v=start_vs[j], point=transforms[i][3])
# newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)

        
#         res_ind = [r.index for r in residues]
#         atoms1 = list(residues[index1].atoms())
#         atoms2 = list(residues[index2].atoms())
#         atoms3 = list(residues[index3].atoms())
#         Natom_ind = [a.index for a in atoms1+atoms2+atoms3 if a.name == 'N']
#         CAatom_ind = [a.index for a in atoms1+atoms2+atoms3 if a.name == 'CA']
#         Catom_ind = [a.index for a in atoms1+atoms2+atoms3 if a.name == 'C']

#         Natom_vecs = [numpy.array([v._value for v in positions[x]]) for x in Natom_ind]
#         CAatom_vecs = [numpy.array([v._value for v in positions[x]]) for x in CAatom_ind]
#         Catom_vecs = [numpy.array([v._value for v in positions[x]]) for x in Catom_ind]
    
        
        
        
        #pick index
        #identify next residue- index +1 or wrapped version
        # generate quaternion
        # apply quaternion

        
    def permute(self, positions, movement):
        # maybe split into a 'copy' function and a control loop
        # add checking for d vs l amino acids and different target vectors
        idx = [x for x in range(len(self.seq))]
        new_idx = idx[movement:] + idx[:movement]
        
        start_chir = numpy.array([1 if x.isupper() else -1 for x in self.seq])
        end_chir = numpy.array([1 if x.isupper() else -1 for x in self.seq[movement:]+self.seq[:movement]])
        chir_map = start_chir * end_chir #-1 if chirality is flipped, 1 if preserved
        
        reslist = [r for r in self.model.topology.residues()]
        ca_ids = []
        cb_ids = []
        ha_ids = []
        ca_cb_at_ids = []
        bb_at_ids = []
        sc_at_ids = []
        #get the indices of the atoms for permuting. Should actually only have to do this once.
        for res in reslist:
            ca_id = [a.index for a in res.atoms() if a.name == 'CA']
            ca_ids.append(ca_id[0])
            cb_id = [a.index for a in res.atoms() if a.name in ('CB', 'HA3')]
            cb_ids.append(cb_id[0])
            ca_cb_at_ids.append(ca_id + cb_id)
            n_id = [a.index for a in res.atoms() if a.name == 'N']
            if res.name == 'PRO':
                h_id = [a.index for a in res.atoms() if a.name == 'CD']
            else: 
                h_id = [a.index for a in res.atoms() if a.name == 'H']
            ha_id = [a.index for a in res.atoms() if a.name in ('HA', 'HA2')]
            ha_ids.append(ha_id[0])
            c_id = [a.index for a in res.atoms() if a.name == 'C']
            o_id = [a.index for a in res.atoms() if a.name == 'O']
#             bb_at_ids.append(n_id + h_id + ca_id + ha_id + c_id + o_id)
            bb_at_ids.append(n_id + h_id + ca_id + c_id + o_id)
            if res.name == 'PRO':
                sc_ids = [a.index for a in res.atoms() if a.name not in ('N', 'CD', 'HA', 'HA2', 'CA', 'C', 'O')]
            else:
                sc_ids = [a.index for a in res.atoms() if a.name not in ('N', 'H', 'HA', 'HA2', 'CA', 'C', 'O')]    

            sc_at_ids.append(sc_ids)

        start_positions = copy.deepcopy(positions)    
        end_positions = copy.deepcopy(start_positions)
        
        #ca-cb vectors for side chains to begin with. Will always be the 'true' side chain
        # so will always be ca-cb(ha1)
        start_vectors = [[start_positions[x[0]]._value, start_positions[x[1]]._value] for x in ca_cb_at_ids]
        #destination vectors- ca-cb(ha1) if chirality to be preserved, else ca-ha(ha2) vector
        end_vectors = []
        for i in range(len(chir_map)):
            if chir_map[i] == 1:
                # preserve chirality so copy ca-cb to ca-cb, regardless of d or l
                end_vectors.append([start_positions[ca_ids[i]]._value, start_positions[cb_ids[i]]._value])
            elif chir_map[i] == -1:
                # flip the chirality, so destination vector is now ca-ha(ha2)
                end_vectors.append([start_positions[ca_ids[i]]._value, start_positions[ha_ids[i]]._value])
            
        transforms = []
        for i in idx:

            t = ampal.geometry.find_transformations(start_vectors[new_idx[i]][0], start_vectors[new_idx[i]][1],
                                                    end_vectors[i][0], end_vectors[i][1])            
            transforms.append(t)

        for i in idx:
            #apply rotations to side chains atom vectors
            start_vs = [numpy.array([x._value for x in start_positions[y]]) for y in sc_at_ids[new_idx[i]]]

            quat = Quaternion.angle_and_axis(angle=transforms[i][1], axis=transforms[i][2])
            for j in range(len(sc_at_ids[new_idx[i]])):
                rotated_vec = quat.rotate_vector(v=start_vs[j], point=transforms[i][3])
                rotated_vec += transforms[i][0]        
                newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
                end_positions[sc_at_ids[new_idx[i]][j]] = newpos
            #copy backbone atom vectors
            # including Ha here doesn't work as it will always overlay with existing Ha
            for j in range(len(bb_at_ids[0])):
                end_positions[bb_at_ids[new_idx[i]][j]] = start_positions[bb_at_ids[i][j]]
            if chir_map[i] == 1:
                end_positions[ha_ids[new_idx[i]]] = start_positions[ha_ids[i]]
            elif chir_map[i] == -1:
                end_positions[ha_ids[new_idx[i]]] = start_positions[cb_ids[i]]
            
            
        return end_positions
    
    
    def check_chirality(self, res_index, positions):
        """
        Take in an index and the positions, and report whether it is D or L
        
        Returns D, L (filter glycines out before this)
        
        dihe should be -120ish for L amino acids, +120ish for D
        """
        res = [r for r in self.model.topology.residues()][res_index]
# should work for glycine now to preserve prochirality of the hydrogens
        ca_id = next(a.index for a in res.atoms() if a.name == 'CA')
        cb_id = next(a.index for a in res.atoms() if a.name in ('CB', 'HA3'))
#         c_id = [a.index for a in res.atoms() if a.name == 'C'][0]
        n_id = next(a.index for a in res.atoms() if a.name == 'N')
        ha_id = next(a.index for a in res.atoms() if a.name in ('HA', 'HA2'))

        ca_v = positions[ca_id]._value
        cb_v = positions[cb_id]._value
#         c_v = positions[c_id]._value
        n_v = positions[n_id]._value
        ha_v = positions[ha_id]._value

        dihe = dihedral(cb_v, n_v, ca_v, ha_v)
        
        if dihe < 0:
            return('L')
        else:
            return('D')

        # return conditions are very loose here- relying on model to not do anything mad
    
    
    def invert_chirality(self, res_index, positions):
        """
        Take a position and index and swap Ha and side chain positions
        """
        outpos = copy.deepcopy(positions)
        res = [r for r in self.model.topology.residues()][res_index]
        ca_id = next(a.index for a in res.atoms() if a.name == 'CA')
        # should work for glycines as well now
        ha_id = next(a.index for a in res.atoms() if a.name in ('HA', 'HA2'))
        
        cb_id = next(a.index for a in res.atoms() if a.name in ('CB', 'HA3'))

        ca_v = positions[ca_id]._value
        ha_v = positions[ha_id]._value
        cb_v = positions[cb_id]._value


        
#         ca_ha_vec = ha_v - ca_v
#         ca_cb_vec = cb_v - ca_v
        
        #transformation to move SC to Ha position
        # separate one for each element of side chain???
        t1 = ampal.geometry.find_transformations(ca_v, cb_v, ca_v, ha_v)
        
        #transformation to move Ha to SC position
        t2 = ampal.geometry.find_transformations(ca_v, ha_v, ca_v, cb_v)
        
        
        q1 = Quaternion.angle_and_axis(angle=t1[1], axis=t1[2])
        
        # don't need this as the Ha doesn't need rotated?
        q2 = Quaternion.angle_and_axis(angle=t2[1], axis=t2[2])
                
        if res.name == 'PRO':
            sc_ids = [a.index for a in res.atoms() if a.name not in ('N', 'HA', 'HA2', 'CA', 'C', 'O')]
        else:
            sc_ids = [a.index for a in res.atoms() if a.name not in ('N', 'H', 'HA', 'HA2', 'CA', 'C', 'O')]
        
        new_ha_v = q2.rotate_vector(v=ha_v, point=t2[3])
        new_ha_v2 = new_ha_v + t2[0]
        


        new_ha_pos = unit.quantity.Quantity(vec3.Vec3(*[x for x in new_ha_v2]), unit=unit.nanometer)
        outpos[ha_id] = new_ha_pos
        
#         ha_newpos = ha_v - (2*ca_ha_vec) # check this with simple 2D example
#         newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in ha_newpos]), unit=unit.nanometer)
#         positions[ha_id] = newpos
        
        sc_pos = [positions[_]._value for _ in sc_ids]
#         sc_newpos = copy.deepcopy(sc_pos)

        #rotate and translate SC
        for j in range(len(sc_ids)):
            rotated_vec = q1.rotate_vector(v=sc_pos[j], point=t1[3])
            rotated_vec += t1[0]
            newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
#             sc_newpos[j] = newpos
            outpos[sc_ids[j]] = newpos
        
        
        
#         rotated_vec = quat.rotate_vector(v=start_vs[j], point=transforms[i][3])
#                 rotated_vec += transforms[i][0]        
#                 newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
#                 end_positions[sc_at_ids[new_idx[i]][j]] = newpos
            
#             quat = Quaternion.angle_and_axis(angle=transforms[i][1], axis=transforms[i][2])
#             for j in range(len(sc_at_ids[new_idx[i]])):
#                 rotated_vec = quat.rotate_vector(v=start_vs[j], point=transforms[i][3])
#                 rotated_vec += transforms[i][0]        
#                 newpos = unit.quantity.Quantity(vec3.Vec3(*[x for x in rotated_vec]), unit=unit.nanometer)
#                 end_positions[sc_at_ids[new_idx[i]][j]] = newpos   
            


        return(outpos)
    
    def sc_chir_check_flip(self, positions, flip=True):
        res = [r for r in self.model.topology.residues()]
        ind = []
        for i, r in enumerate(res):
            if r.name in ['THR', 'ILE']:
                ind.append(i)
        
        wrong_chir = [] # count of number of wrong chirality side chains
        for i in ind:
            try:
                a1 = next(a.index for a in res[i].atoms() if a.name == 'HB')
                a2 = next(a.index for a in res[i].atoms() if a.name == 'CA')
                a3 = next(a.index for a in res[i].atoms() if a.name in ('CB', 'HA3'))
                cg_candidates = [a.index for a in res[i].atoms() if a.name in ('CG1', 'CG2')]
                if not cg_candidates: continue
                cg_id = cg_candidates[0]
                
                a1_v = positions[a1]._value
                a2_v = positions[a2]._value
                a3_v = positions[a3]._value
                cg_v = positions[cg_id]._value
                dihe = dihedral(a1_v, a2_v, a3_v, cg_v)
                
                is_d = self.seq[i].islower()
                
                if res[i].name == 'THR':
                    if (not is_d and dihe < 0) or (is_d and dihe > 0):
                        wrong_chir.append(i)
                elif res[i].name == 'ILE' and ((not is_d and dihe > 0) or (is_d and dihe < 0)):
                    wrong_chir.append(i)
            except IndexError:
                continue
                
        if not flip:
            return len(wrong_chir) == 0
            
        for i in wrong_chir:
            try:
                ca_id = next(a.index for a in res[i].atoms() if a.name == 'CA')
                cb_id = next(a.index for a in res[i].atoms() if a.name in ('CB', 'HA3'))
                n_id = next(a.index for a in res[i].atoms() if a.name == 'N')
                
                ca_v = numpy.array(positions[ca_id]._value)
                cb_v = numpy.array(positions[cb_id]._value)
                n_v = numpy.array(positions[n_id]._value)
                
                # Create a plane containing CA, CB, and N
                ca_cb = cb_v - ca_v
                ca_n = n_v - ca_v
                plane_normal = numpy.cross(ca_cb, ca_n)
                plane_normal = plane_normal / numpy.linalg.norm(plane_normal)
                
                # Mirror all atoms beyond CB
                sc_atoms = [a.index for a in res[i].atoms() if a.name not in ('N', 'H', 'CA', 'C', 'O', 'HA', 'HA2', 'CB')]
                
                for a_idx in sc_atoms:
                    atom_v = numpy.array(positions[a_idx]._value)
                    v = atom_v - cb_v
                    dist = numpy.dot(v, plane_normal)
                    mirrored = atom_v - 2 * dist * plane_normal
                    positions[a_idx] = unit.quantity.Quantity(vec3.Vec3(*mirrored), unit=unit.nanometer)
            except IndexError:
                continue
            
        return positions

    def filter_by_rama_rmsd(self, population, rmsd_val):
        #takes a population and keeps best, and only subsequent models if they pass rama cutoff test
        ramalist = []
        for pos in population:
            n_pos = [pos[1][j]._value for j in self.n_indices]
            ca_pos = [pos[1][j]._value for j in self.ca_indices]
            c_pos = [pos[1][j]._value for j in self.c_indices]
            temp_ramas = []
            for k in range(len(self.n_indices)):
                phi = dihedral(c_pos[k-2], n_pos[k-1], ca_pos[k-1], c_pos[k-1])
                psi = dihedral(n_pos[k-1], ca_pos[k-1], c_pos[k-1], n_pos[k])
                temp_ramas.append((phi, psi))

            rama = temp_ramas[1:]+[temp_ramas[0]]
            ramalist.append(rama)

        tmp_current = []
        accepted_indices = []
        for j in range(len(population)):
            is_distinct = True
            for k in accepted_indices:
                rmsd = calc_rmsd2(ramalist[j], ramalist[k])
                if rmsd <= rmsd_val:
                    is_distinct = False
                    break
            if is_distinct:
                tmp_current.append(population[j])
                accepted_indices.append(j)
        return tmp_current

    def optimise(self, n_iter, wp_len=20, samplesize=200, hof_len=5, rama_rmsd=15, n_permute=3, n_flip=5, tol=5, max_iter=100, plot=True):
        #run the optimisation
        
        print(f'optimising sequence {self.seq}')
        self.energies = [self.initial_state[0]]
        self.working_pop = [copy.deepcopy(self.initial_state)]
        self.halloffame = [copy.deepcopy(self.initial_state)]        
        self.rama_rmsd = rama_rmsd
        self.tol = tol
        self.max_iter = max_iter
        
        for i in range(n_iter):
            current_models = []
            self.newpositions = []

            while len(self.newpositions) < (samplesize):
#                 self.startpos = random.choice(self.newpositions)
                self.startpos = random.choice(self.working_pop)[1]               
                id_tmp = sorted(random.sample(range(len(self.seq)), 3))
                ran_int = random.randint(0, 2)
                indices = (id_tmp[ran_int:]+id_tmp[:ran_int])
                self.newpos = find_alternate_positions(self.model, self.startpos, *indices)
                self.newpositions += self.newpos
                
            for j in range(n_permute):
                self.startpos = random.choice(self.working_pop)[1]
#                 self.startpos = random.choice(self.working_pop)               
                move = random.choice(range(1, len(self.seq)-1))
                self.newpos = [self.permute(self.startpos, move)]
                self.newpositions += self.newpos
            
            for j in range(n_flip):
                self.startpos = random.choice(self.working_pop)[1]
                bond = random.choice(range(1, len(self.seq)-1))
                angle = random.random()*360
                self.newpos = [self.flip_amide(self.startpos, bond, angle)]
                self.newpositions += self.newpos
            
#             self.newpositions += [x[1] for x in self.working_pop]
#             self.newpositions += [x[1] for x in self.halloffame]

            # seems silly, but intention is for models in working pop to undergo multiple energy min steps 
           
            for pos in self.newpositions:
        # try/except to protect from particle position is nan error or other openmm issues
                try:
#                 maybe lose this first minimization for speed purposes?
#                 then would need to always do the second one
                    self.simulation.context.setPositions(pos)
                    for k in range(self.force.getNumParticles()):
                        idx = self.force.getParticleParameters(k)[0]
                        self.force.setParticleParameters(k, idx, pos[idx]._value)
                    self.force.updateParametersInContext(self.simulation.context)
                    self.simulation.minimizeEnergy(tolerance=self.tol*unit.kilojoule/(unit.nanometer*unit.mole), maxIterations=self.max_iter)
                    state = self.simulation.context.getState(getEnergy=True, getPositions=True)
                    current_energy = state.getPotentialEnergy() / unit.kilojoules_per_mole
                    current_positions = state.getPositions()
                    
                    flipnow = 0
                    if any(x in 'TtIi' for x in self.seq):
                        current_positions = self.sc_chir_check_flip(current_positions, flip=True)
                        
                    for k in range(len(self.seq)):
                        chir = self.check_chirality(k, current_positions)
                        if (chir == 'L') and (str.islower(self.seq[k])) or (chir == 'D') and not (str.islower(self.seq[k])):
                            current_positions = self.invert_chirality(k, current_positions)
                            self.flip += 1
                            flipnow += 1

                    self.simulation.context.setPositions(current_positions)
                    for k in range(self.force.getNumParticles()):
                        idx = self.force.getParticleParameters(k)[0]
                        self.force.setParticleParameters(k, idx, current_positions[idx]._value)
                    self.force.updateParametersInContext(self.simulation.context)
#                     self.simulation.minimizeEnergy(maxIterations=100)
                    self.simulation.minimizeEnergy(tolerance=self.tol*unit.kilojoule/(unit.nanometer*unit.mole), maxIterations=self.max_iter)                    
                    state = self.simulation.context.getState(getEnergy=True, getPositions=True)
                    current_energy = state.getPotentialEnergy() / unit.kilojoules_per_mole
                    current_positions = state.getPositions()
                    
                    # check again and kill any repeat offenders
                    
                    #add one for every fail
                    badchir = 0
                    
                    if any(x in 'TtIi' for x in self.seq) and not self.sc_chir_check_flip(current_positions, flip=False):
                        badchir += 1
                        # if duff side chain get false out of this, need to kill model and move to next
                        # maybe just add a flag and pick up at cis check?
                        
                    for k in range(len(self.seq)):
#                        if residues[k].name != 'GLY':
                        chir = self.check_chirality(k, current_positions)
                        if (chir == 'L') and (str.islower(self.seq[k])) or (chir == 'D') and not (str.islower(self.seq[k])):
                            badchir += 1
                        # now check for non pro cis residues
                    dihe2 = []
                    reslist = [r for r in self.model.topology.residues()]
                    idx = [x for x in range(len(self.seq))]
                    idx2 = [idx[-1]]+idx+[idx[0]]
                    for j in idx:
                        res = reslist[j]
                        atoms1 = [a for a in res.atoms() if a.name in ('N', 'CA')]
                        atoms2 = [a for a in reslist[idx2[j]].atoms() if a.name in ('C', 'CA')]
                        atoms = atoms2 + atoms1
                        at_idx = [a.index for a in atoms]
                        amides = [current_positions[x]._value for x in at_idx]
                        dihe2.append(dihedral(*amides))
                    if all(abs(d) > 20 for d in dihe2):
                        #attempt to shut down exploding simulations -3000 arbitrary. Scale to seq len?
                        if abs(current_energy) < 400 * len(self.seq) and badchir == 0:
                            new_entry = (current_energy, current_positions)
                            current_models.append(new_entry)
                            self.good += 1
                    else:
                        # there is a non PRO cis amide, don't keep it, count the failure
                        self.cis_bad += 1
                # exception catching for openmm issues- not really needed but prevents problems...
                #... being hidden by try above (try is good but only if except is informative)
                except Exception:
                    pass
            #combine all models and sort by score
            current_models.sort(key = lambda x: x[0])
            
            # Record length before filtering to calculate acceptance rate
            pre_filter_len = len(self.halloffame) + len(current_models)
            
            self.halloffame += current_models
            self.halloffame.sort(key = lambda x: x[0])
            self.halloffame = self.filter_by_rama_rmsd(self.halloffame, self.rama_rmsd)
            
            # Calculate what percentage of models survived the diversity filter
            survival_rate = len(self.halloffame) / pre_filter_len if pre_filter_len > 0 else 1.0
            target_survival = hof_len / pre_filter_len if pre_filter_len > 0 else 0.05
            
            if survival_rate < target_survival:
                # Too strict: fewer models survived than we want to keep in the HoF
                ratio = target_survival / max(0.001, survival_rate)
                step = min(10.0, max(1.0, ratio))
                self.rama_rmsd = max(0.1, self.rama_rmsd - step)
            elif survival_rate > target_survival * 3:
                # Too lax: far more models survived than we need
                ratio = survival_rate / (target_survival * 3)
                step = min(10.0, max(1.0, ratio * 2))
                self.rama_rmsd += step
            
            self.halloffame = self.halloffame[:hof_len]
            # cut to length anyways
            self.energies.append(self.halloffame[0][0])

            if i%5 == 0:
                self.working_pop = self.halloffame[:]
            else:
                self.working_pop += current_models
                self.working_pop.sort(key = lambda x: x[0])

            sys.stdout.write(f"\r HOF at iter {i} is {[x[0] for x in self.halloffame]} best {self.working_pop[0][0]} worst {self.working_pop[-1][0]} rama {self.rama_rmsd}"+" "*100)
            sys.stdout.flush()

        #plot stuff, obviously
        if plot:
            plt.plot(range(n_iter+1), self.energies)
            plt.title('optimisation curve')
            plt.show()

    def plot_halloffame_ramachandran(self, cols=3):
        import math

        import matplotlib.pyplot as plt
        import numpy
        
        # 1. Extract Dihedrals (using the existing logic)
        ramalist = []
        for pos in self.halloffame:
            n_pos = [pos[1][j]._value for j in self.n_indices]
            ca_pos = [pos[1][j]._value for j in self.ca_indices]
            c_pos = [pos[1][j]._value for j in self.c_indices]
            temp_ramas = []
            for k in range(len(self.n_indices)):
                phi = dihedral(c_pos[k-2], n_pos[k-1], ca_pos[k-1], c_pos[k-1])
                psi = dihedral(n_pos[k-1], ca_pos[k-1], c_pos[k-1], n_pos[k])
                temp_ramas.append((phi, psi))
            rama = temp_ramas[1:]+[temp_ramas[0]]
            ramalist.append(rama)

        # 2. Setup the Grid
        n_plots = len(ramalist)
        if n_plots == 0:
            print("No models in Hall of Fame to plot.")
            return

        rows = math.ceil(n_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        
        # Ensure axes is a flat array for easy indexing even if it's 1x1 or 1xN
        if n_plots == 1 and rows == 1 and cols == 1: 
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else: 
            axes = axes.flatten()

        colours = plt.cm.rainbow([x for x in numpy.linspace(0, 1, len(self.seq))])

        # 3. Plot Each HoF Entry
        for j in range(n_plots):
            ax = axes[j]
            ax.set_aspect('equal')
            ax.set_xlim([-180, 180])
            ax.set_ylim([-180, 180])
            
            # Add gridlines for standard Ramachandran quadrants
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

            xser = [ramalist[j][x][0] for x in range(len(self.seq))]
            yser = [ramalist[j][y][1] for y in range(len(self.seq))]

            ax.set_xlabel('phi', size=12)
            ax.set_ylabel('psi', size=12)
            ax.scatter(xser, yser, c=colours, s=60, edgecolor='black', zorder=5)
            
            # Truncate score for cleaner title
            ax.set_title(f'Rank {j+1} (Score: {self.halloffame[j][0]:.1f})', size=12)

            for k in range(len(self.seq)):
                ax.annotate(f"{self.seq[k]}{k}", (xser[k]+5, yser[k]+5), size=9, zorder=10)

        # 4. Hide unused subplots (if n_plots isn't a multiple of cols)
        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()



    def export_halloffame_pdb(self, filename):
        """Exports the entire hall of fame as a single multi-model PDB file."""
        from openmm.app import PDBFile
        with open(filename, 'w') as f:
            PDBFile.writeHeader(self.model.topology, file=f)
            for i, (energy, positions) in enumerate(self.halloffame):
                PDBFile.writeModel(self.model.topology, positions, file=f, modelIndex=i+1)
            PDBFile.writeFooter(self.model.topology, file=f)
