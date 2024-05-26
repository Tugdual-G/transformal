#!/usr/bin/env python3
from matplotlib.pyplot import axis
import numpy as np
import trimesh
import networkx as nx

def create_dirac_op(trimesh):
    faces_edges_idx = trimesh.faces_unique_edges
    edges = trimesh.edges_unique[faces_edges_idx]
    D_block = np.zeros((trimesh.faces.shape[0]*4,trimesh.vertices.shape[0]*4))
    # block = np.zeros((4,4))

    for i in range(faces_edges_idx.shape[0]):
        graph = nx.from_edgelist(edges[i])
        vertex_ordered = np.array(nx.cycle_basis(graph)[0][::-1], dtype=np.int16)
        v = trimesh.vertices[vertex_ordered]
        area_x2 = abs(np.linalg.norm(np.cross(v[1]-v[0],v[2]-v[1])))
        for j in range(3):
            e_j = v[(j+2)%3]-v[(j+1)%3]
            Dij = -(e_j)/area_x2
            block = np.array([[0, -Dij[0], -Dij[1], -Dij[2]],
                                [Dij[0],  0, -Dij[2],  Dij[1]],
                                [Dij[1],  Dij[2],  0, -Dij[0]],
                                [Dij[2], -Dij[1],  Dij[0],  0]])

            D_block[i*4:i*4+4,vertex_ordered[j]*4:vertex_ordered[j]*4+4] = block
    return D_block

def create_R(trimesh, rho):
    nf = trimesh.faces.shape[0]
    nv = trimesh.vertices.shape[0]
    P = np.zeros((nf*4,nf*4))
    for i in range(nf):
        for k in range(4):
            P[i*4+k, i*4+k] = rho[i]

    B = np.zeros((nf*4,nv*4))

    faces_edges_idx = trimesh.faces_unique_edges
    edges = trimesh.edges_unique[faces_edges_idx]

    for i in range(nf):
        graph = nx.from_edgelist(edges[i])
        vertex_ordered = np.array(nx.cycle_basis(graph)[0][::-1], dtype=np.int16)
        for j in vertex_ordered:
            for k in range(4):
                B[i*4+k,j*4+k] = 1.0/3

    return P @ B


def create_Mf(trimesh):
    nf = trimesh.faces.shape[0]
    Mf = np.zeros((nf*4,nf*4))
    for i in range(nf):
        area = trimesh.area_faces[i]
        for k in range(4):
            Mf[i*4+k,i*4+k] = area
    return Mf


def create_Mv(trimesh):
    nv = trimesh.vertices.shape[0]
    Mv = np.zeros((nv*4,nv*4))
    for i in range(nv):
        faces = trimesh.vertex_faces[i]
        area = trimesh.area_faces[faces[faces != -1]]
        area = np.sum(area, axis=0)/3
        for k in range(4):
            Mv[i*4+k,i*4+k] = area
    return Mv

def mul_quatern(u, v):
    w0, x0, y0, z0 = u
    w1, x1, y1, z1 = v
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def new_edges(trimesh, lambd):
    # TODO check rotation order
    assert lambd.shape[0]%4 == 0
    nv = trimesh.vertices.shape[0]
    edges = trimesh.edges_unique
    edge_vects = np.zeros((nv*4,nv*4), dtype=np.float64)
    eij = np.zeros(4, dtype=np.float64)
    lambdc = lambd.copy()

    for i in range(lambd.shape[0]//4):
        lambdc[4*i+1:4*i+4] *= -1

    for i in range(edges.shape[0]):
        vi = edges[i,0]
        vj = edges[i,1]
        eij[1:] = trimesh.vertices[vj]-trimesh.vertices[vi]
        e_new  = 1/3 * mul_quatern(lambdc[vi*4:vi*4+4], mul_quatern(eij, lambd[vi*4:vi*4+4]))
        e_new += 1/6 * mul_quatern(lambdc[vi*4:vi*4+4], mul_quatern(eij, lambd[vj*4:vj*4+4]))
        e_new += 1/6 * mul_quatern(lambdc[vj*4:vj*4+4], mul_quatern(eij, lambd[vi*4:vi*4+4]))
        e_new += 1/3 * mul_quatern(lambdc[vj*4:vj*4+4], mul_quatern(eij, lambd[vj*4:vj*4+4]))

        block = np.array([[0.0, -e_new[1], -e_new[2], -e_new[3]],
                          [e_new[1],  0.0, -e_new[3],  e_new[2]],
                          [e_new[2],  e_new[3],  0.0, -e_new[1]],
                          [e_new[3], -e_new[2],  e_new[1],  0.0]])

        edge_vects[vi*4:vi*4+4, vj*4:vj*4+4] = block
        edge_vects[vj*4:vj*4+4, vi*4:vi*4+4] = -block

    return edge_vects


def new_edges_divergence(trimesh, lambd):
    nv = trimesh.vertices.shape[0]

    g = nx.from_edgelist(trimesh.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(nv)]
    one_ordered = [nx.cycle_basis(g.subgraph(i)) for i in one_ring]

    eij = np.zeros(4, dtype=np.float64)
    lambdc = lambd.copy()

    for i in range(lambd.shape[0]//4):
        lambdc[4*i+1:4*i+4] *= -1

    div = np.zeros((nv*4))
    for i in range(nv):
        if len(one_ordered[i]) >0:
            ring_vert = trimesh.vertices[one_ordered[i][0]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert-trimesh.vertices[i,:]

            nv = ring_vert.shape[0]

            # area of the ring
            A = np.sum(np.linalg.norm(np.cross(edges_vect[:-1],edges_vect[1:]), axis=1))

            for j in range(nv):
                e1 = edges_vect[(j-1)%nv]
                # e1 /= np.linalg.norm(e1)
                o1 =  edges_vect[j]-e1
                # o1 /= np.linalg.norm(o1)
                cos1 = np.dot(e1,o1)
                sin1 = np.linalg.norm(np.cross(o1,e1))
                if sin1 != 0.0 :
                    cot1 = cos1/sin1
                else :
                    cot1 = 0

                e2 = edges_vect[(j+1)%nv]
                # e2 /= np.linalg.norm(e2)
                o2 =  edges_vect[j]-e2
                # o2 /= np.linalg.norm(o2)
                cos2 = np.dot(e2,o2)
                sin2 = np.linalg.norm(np.cross(e2,o2))

                if sin2 != 0.0 :
                    cot2 = cos2/sin2
                else :
                    cot2 = 0

                vj = one_ordered[i][0][j]
                eij[1:] = edges_vect[j]
                e_new  = 1/3 * mul_quatern(lambdc[i*4:i*4+4], mul_quatern(eij, lambd[i*4:i*4+4]))
                e_new += 1/6 * mul_quatern(lambdc[i*4:i*4+4], mul_quatern(eij, lambd[vj*4:vj*4+4]))
                e_new += 1/6 * mul_quatern(lambdc[vj*4:vj*4+4], mul_quatern(eij, lambd[i*4:i*4+4]))
                e_new += 1/3 * mul_quatern(lambdc[vj*4:vj*4+4], mul_quatern(eij, lambd[vj*4:vj*4+4]))
                # print(np.linalg.norm(e_new-eij))

                div[4*i+1:4*i+4] -= e_new[1:]*(cot2+cot1)
            # div[4*i+1:4*i+4] /= 2*A
    return div


def quaternionic_laplacian_matrix(trimesh):
    nv = trimesh.vertices.shape[0]
    g = nx.from_edgelist(trimesh.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(nv)]
    one_ordered = [nx.cycle_basis(g.subgraph(i)) for i in one_ring]

    L = np.zeros((nv*4,nv*4))
    # area matrix
    A = np.zeros((nv*4, nv*4))

    for i in range(nv):
        if len(one_ordered[i]) >0:
            ring_vert = trimesh.vertices[one_ordered[i][0]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert-trimesh.vertices[i,:]

            # area of the ring
            a = np.sum(np.linalg.norm(np.cross(edges_vect[:-1],edges_vect[1:]), axis=1))

            nv = ring_vert.shape[0]
            for k in range(nv):
                e1 = edges_vect[(k-1)%nv].copy()
                # e1 /= np.linalg.norm(e1)
                o1 = ring_vert[k]-ring_vert[(k-1)%nv]
                # o1 /= np.linalg.norm(o1)
                cos1 = np.dot(e1,o1)
                sin1 = np.linalg.norm(np.cross(o1,e1))
                if sin1 != 0.0 :
                    cot1 = cos1/sin1
                else :
                    cot1 = 0

                e2 = edges_vect[(k+1)%nv].copy()
                # e2 /= np.linalg.norm(e2)
                o2 = ring_vert[k] - ring_vert[(k+1)%nv]
                # o2 /= np.linalg.norm(o2)
                cos2 = np.dot(e2,o2)
                sin2 = np.linalg.norm(np.cross(e2,o2))

                if sin2 != 0.0 :
                    cot2 = cos2/sin2
                else :
                    cot2 = 0

                j = one_ordered[i][0][k]
                for l in range(4):
                    L[4*i+l,4*j+l] = (cot2+cot1)#/(2*a)
                    L[4*i+l,4*i+l] -= (cot2+cot1)#/(2*a)

            for l in range(4):
                A[4*i+l,4*i+l] = 2*a
    return L, A

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d
    from trimesh_curvature import mean_curvature
    from scipy.linalg import eigh
    from scipy.linalg import solve, issymmetric


    trimesh = trimesh.load('knobitle.ply')

    L, Area = quaternionic_laplacian_matrix(trimesh)


    Ainv = Area.copy()
    for i in range(Area.shape[0]):
        if Ainv[i,i] != 0:
            Ainv[i,i] =  1/Ainv[i,i]

    nv = trimesh.vertices.shape[0]
    kN = mean_curvature(trimesh)
    k = np.linalg.norm(kN, axis=1)
    face_k = k[trimesh.faces]
    face_k = np.mean(face_k, axis=1)
    rho = -0.01*face_k

    D = create_dirac_op(trimesh)
    R = create_R(trimesh, rho)

    A = D - R
    Mv = create_Mv(trimesh)
    Mv_inv = Mv.copy()
    for i in range(Mv.shape[0]):
        Mv_inv[i,i] = 1/Mv_inv[i,i]

    Mf = create_Mf(trimesh)

    V = Mv.copy()
    for i in range(Mv.shape[0]):
        V[i,i] = 1/np.sqrt(V[i,i])

    AV_str = Mv_inv @ (A @ V).T @ Mf

    eigvals, eigvecs = eigh(AV_str @ (A @ V), eigvals_only=False, subset_by_index=[0,0])
    lambd = V @ eigvecs[:,0]
    lambd.shape = (nv, 4)
    lambd /= np.mean(np.linalg.norm(lambd,axis=1))
    print(lambd)
    lambd.shape = (nv*4, )

    div_e = new_edges_divergence(trimesh, lambd)



    print(f"{issymmetric(L) = }")
    new_vertices = solve(L, div_e,assume_a='sym')
    residual = (L@new_vertices - div_e)
    print(f"{np.linalg.norm(residual)=}")
    new_vertices.shape = (nv,4)

    trimesh.vertices[:,:] = new_vertices[:,1:]


    # vertices = np.zeros(nv*4)
    # vertices[1::4] = trimesh.vertices[:,0]
    # vertices[2::4] = trimesh.vertices[:,1]
    # vertices[3::4] = trimesh.vertices[:,2]

    # vertices.shape = (nv*4,1)
    # L, A = quaternionic_laplacian_matrix(trimesh)
    # plt.spy(L)
    # plt.show()
    # Ainv = A.copy()
    # for i in range(A.shape[0]):
    #     if Ainv[i,i] != 0:
    #         Ainv[i,i] =  1/Ainv[i,i]

    # kN = Ainv @ L @ vertices
    # kN.shape = (nv,4)
    # kN = kN[:,1:]



    kN = mean_curvature(trimesh)
    vertex_normals = trimesh.vertex_normals
    k = np.sum(kN*vertex_normals, axis=1)
    face_k = k[trimesh.faces]
    face_k = np.mean(face_k, axis=1)
    k_norm = (face_k-face_k.min())/np.ptp(face_k)
    cm = plt.cm.plasma(k_norm)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # for i in range(kN.shape[0]):
    #     normalsegm = np.stack((trimesh.vertices[i],trimesh.vertices[i,:]-20*kN[i]/np.linalg.norm(kN[i])))
    #     ax.plot(normalsegm[:,0],normalsegm[:,1],normalsegm[:,2],'b')

    triangles = trimesh.vertices[trimesh.faces]
    pc = art3d.Poly3DCollection(triangles, facecolors=cm,shade=True, edgecolors=(1,1,1,0.2), alpha=0.9)
    ax.add_collection(pc)

    xlim = trimesh.vertices[:,0].min(), trimesh.vertices[:,0].max()
    ylim = trimesh.vertices[:,1].min(), trimesh.vertices[:,1].max()
    zlim = trimesh.vertices[:,2].min(), trimesh.vertices[:,2].max()

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([1,1,1])
    plt.show()
