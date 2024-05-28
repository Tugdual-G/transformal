#!/usr/bin/env python3
from matplotlib.pyplot import axis
import numpy as np
import trimesh
import networkx as nx
from scipy.sparse.linalg import spsolve, norm, inv
from scipy.sparse import csc_array
from scipy.linalg import issymmetric, eig, eigh, solve, expm_cond

def create_dirac_op_fv(trimesh):
    faces_edges_idx = trimesh.faces_unique_edges
    edges = trimesh.edges_unique[faces_edges_idx]
    D_block = np.zeros((trimesh.faces.shape[0]*4,trimesh.vertices.shape[0]*4))
    # block = np.zeros((4,4))

    # for each triangle
    for i in range(faces_edges_idx.shape[0]):
        graph = nx.from_edgelist(edges[i])
        vertex_ordered_idx = np.array(nx.cycle_basis(graph)[0][::-1], dtype=np.int16)
        v = trimesh.vertices[vertex_ordered_idx]

        vertex_normal = trimesh.vertex_normals[vertex_ordered_idx[0]]
        order_normal = np.cross(v[1]-v[0], v[2]-v[0])
        sign = np.dot(vertex_normal, order_normal)
        sign = int(np.sign(sign))
        vertex_ordered_idx = vertex_ordered_idx[::sign]
        v = v[::sign,:]
        #
        area_x2 = np.linalg.norm(np.cross(v[1]-v[0],v[2]-v[1]))
        for j in range(3):
            e_j = v[(j+2)%3]-v[(j+1)%3]
            Dij = -(e_j)/area_x2
            block = np.array([[0, -Dij[0], -Dij[1], -Dij[2]],
                                [Dij[0],  0, -Dij[2],  Dij[1]],
                                [Dij[1],  Dij[2],  0, -Dij[0]],
                                [Dij[2], -Dij[1],  Dij[0],  0]])

            D_block[i*4:i*4+4,vertex_ordered_idx[j]*4:vertex_ordered_idx[j]*4+4] = block
    return D_block

def create_dirac_op_vv(trimesh, rho):
    vertices = trimesh.vertices
    nv = vertices.shape[0]
    faces_edges_idx = trimesh.faces_unique_edges
    edges = trimesh.edges_unique[faces_edges_idx]
    X_block = np.zeros((nv*4,nv*4))
    e_i = np.zeros(4)
    e_j = np.zeros(4)
    assert rho.shape[0] == nv
    block_ij = np.zeros((4,4))

    # for each triangle
    for f_i in range(faces_edges_idx.shape[0]):
        # Get the ordered vertices list of each triangle
        graph = nx.from_edgelist(edges[f_i])
        vertex_ordered_idx = np.array(nx.cycle_basis(graph)[0], dtype=np.int16)
        f_v = trimesh.vertices[vertex_ordered_idx]

        vertex_normal = trimesh.face_normals[f_i]
        order_normal = np.cross(f_v[1]-f_v[0], f_v[2]-f_v[0])
        sign = np.dot(vertex_normal, order_normal)
        sign = int(np.sign(sign))
        vertex_ordered_idx = vertex_ordered_idx[::sign]
        f_v = f_v[::sign,:]
        #
        area_x2 = np.linalg.norm(np.cross(f_v[1]-f_v[0],f_v[2]-f_v[1]))
        for t_i in range(3):
            e_i[1:] = f_v[(t_i+2)%3]-f_v[(t_i+1)%3]
            i = vertex_ordered_idx[t_i]
            for t_j in [(t_i+1)%3,(t_i+2)%3]:
            # for t_j in range(3):
                j = vertex_ordered_idx[t_j]
                e_j[1:] = f_v[(t_j+2)%3]-f_v[(t_j+1)%3]
                X_ij = - mul_quatern(e_i, e_j)/(2*area_x2) + (rho[i]*e_j-rho[j]*e_i)/6.0
                X_ij[0] +=  rho[i]*rho[j]*area_x2/18.0
                block_ij[:,:] = ([[X_ij[0], -X_ij[1], -X_ij[2], -X_ij[3]],
                                  [X_ij[1],  X_ij[0], -X_ij[3],  X_ij[2]],
                                  [X_ij[2],  X_ij[3],  X_ij[0], -X_ij[1]],
                                  [X_ij[3], -X_ij[2],  X_ij[1],  X_ij[0]]])

                X_block[i*4:i*4+4,j*4:j*4+4] += block_ij
    return X_block


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


def new_edges_divergence(trimesh, lambd):
    nv = trimesh.vertices.shape[0]

    g = nx.from_edgelist(trimesh.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(nv)]
    one_ordered = [nx.cycle_basis(g.subgraph(i)) for i in one_ring]

    eij = np.zeros(4, dtype=np.float64)
    lambdc = lambd.copy()

    constraint = [[0,np.zeros(4)],[0,np.zeros(4)]]
    assert lambd.shape[0]%4 == 0
    for i in range(lambd.shape[0]//4):
        lambdc[4*i+1:4*i+4] *= -1

    div = np.zeros((nv*4))
    for i in range(nv):
        if len(one_ordered[i]) >0:
            ring_vert = trimesh.vertices[one_ordered[i][0]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert-trimesh.vertices[i,:]

            vertex_normal = trimesh.vertex_normals[i]
            sign = np.dot(vertex_normal,np.cross(edges_vect[0],edges_vect[1]))
            sign = int(np.sign(sign))
            edges_vect = edges_vect[::sign,:]
            one_ordered[i][0] = one_ordered[i][0][::sign]

            ring_nv = ring_vert.shape[0]

            # area of the ring
            # A = np.sum(np.linalg.norm(np.cross(edges_vect[:-1],edges_vect[1:]), axis=1))
            for k in range(ring_nv):
                e1 = -edges_vect[(k-1)%ring_nv]
                # e1 /= np.linalg.norm(e1)
                o1 =  edges_vect[k]+e1
                # o1 /= np.linalg.norm(o1)
                cos1 = np.dot(e1,o1)
                sin1 = np.linalg.norm(np.cross(o1,e1))
                if sin1 != 0.0 :
                    cot1 = cos1/sin1
                else :
                    cot1 = 0

                e2 = -edges_vect[(k+1)%ring_nv]
                # e2 /= np.linalg.norm(e2)
                o2 =  edges_vect[k]+e2
                # o2 /= np.linalg.norm(o2)
                cos2 = np.dot(e2,o2)
                sin2 = np.linalg.norm(np.cross(e2,o2))

                if sin2 != 0.0 :
                    cot2 = cos2/sin2
                else :
                    cot2 = 0

                j = one_ordered[i][0][k]
                eij[1:] = edges_vect[k]
                e_new  = 1/3 * mul_quatern(lambdc[i*4:i*4+4], mul_quatern(eij, lambd[i*4:i*4+4]))
                e_new += 1/6 * mul_quatern(lambdc[i*4:i*4+4], mul_quatern(eij, lambd[j*4:j*4+4]))
                e_new += 1/6 * mul_quatern(lambdc[j*4:j*4+4], mul_quatern(eij, lambd[i*4:i*4+4]))
                e_new += 1/3 * mul_quatern(lambdc[j*4:j*4+4], mul_quatern(eij, lambd[j*4:j*4+4]))
                # print(np.linalg.norm(e_new-eij))

                div[4*i+1:4*i+4] += e_new[1:]*(cot2+cot1)

    constraint[0][0] = i-1
    constraint[0][1][1:] = trimesh.vertices[i,:]
    constraint[1][0] = j
    constraint[1][1][1:] =  constraint[0][1][1:]+e_new[1:]

    return div, constraint


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
            vertex_normal = trimesh.vertex_normals[i]
            sign = np.dot(vertex_normal,np.cross(edges_vect[0],edges_vect[1]))
            sign = int(np.sign(sign))
            edges_vect = edges_vect[::sign,:]
            one_ordered[i][0] = one_ordered[i][0][::sign]
            ring_vert = ring_vert[::sign,:]

            # area of the ring
            a_x2 = np.sum(np.linalg.norm(np.cross(edges_vect[:-1],edges_vect[1:]), axis=1))

            nv = ring_vert.shape[0]
            # iterating over each of the edges adjacent to the vertex i
            for k in range(nv):
                # e1 = -edges_vect[(k-1)%nv]
                # # e1 /= np.linalg.norm(e1)
                # o1 =  edges_vect[k]+e1
                # # o1 /= np.linalg.norm(o1)
                # cos1 = np.dot(e1,o1)
                # sin1 = np.linalg.norm(np.cross(o1,e1))
                # if sin1 != 0.0 :
                #     cot1 = cos1/sin1
                # else :
                #     cot1 = 0

                # e2 = -edges_vect[(k+1)%nv]
                # # e2 /= np.linalg.norm(e2)
                # o2 =  edges_vect[k]+e2
                # # o2 /= np.linalg.norm(o2)
                # cos2 = np.dot(e2,o2)
                # sin2 = np.linalg.norm(np.cross(e2,o2))

                # if sin2 != 0.0 :
                #     cot2 = cos2/sin2
                # else :
                #     cot2 = 0

                e1 = edges_vect[(k-1)%nv]
                # e1 /= np.linalg.norm(e1)
                o1 = ring_vert[k]-ring_vert[(k-1)%nv]
                # o1 /= np.linalg.norm(o1)
                cos1 = np.dot(e1,o1)
                sin1 = np.linalg.norm(np.cross(o1,e1))
                if sin1 != 0.0 :
                    cot1 = cos1/sin1
                else :
                    cot1 = 0

                e2 = edges_vect[(k+1)%nv]
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
                    L[4*i+l,4*j+l] = (cot2+cot1)#/(2*a_x2)
                    L[4*i+l,4*i+l] -= (cot2+cot1)#/(2*a_x2)

            for l in range(4):
                A[4*i+l,4*i+l] = 2*a_x2
    return L, A

def transform(trimesh, rho):
    L, Area = quaternionic_laplacian_matrix(trimesh)

    # Ainv = Area.copy()
    # for i in range(Area.shape[0]):
    #     if Ainv[i,i] != 0:
    #         Ainv[i,i] =  1/Ainv[i,i]

    vertices = trimesh.vertices
    nv = vertices.shape[0]

    # D = create_dirac_op(trimesh)
    # R = create_R(trimesh, rho)

    # A = D - R
    # Mv = create_Mv(trimesh)
    # Mv_inv = Mv.copy()
    # for i in range(Mv.shape[0]):
    #     Mv_inv[i,i] = 1/Mv_inv[i,i]

    # Mf = create_Mf(trimesh)

    # V = Mv.copy()
    # for i in range(Mv.shape[0]):
    #     V[i,i] = 1/np.sqrt(V[i,i])
    # V /= np.mean(V)

    # AV_str = Mv_inv @ (A @ V).T @ Mf

    # D_Matrix = AV_str @ (A @ V)
    # print(D_Matrix[:4,:4])
    # print(f"{issymmetric(D_Matrix) = }")
    # eigvals, eigvecs = eig(D_Matrix)
    # first_eigvec = np.real(eigvecs[:,np.argmin(np.real(eigvals))])
    X = create_dirac_op_vv(trimesh, rho)
    print(f"{issymmetric(X) = }")
    eigvals, eigvecs = eigh(X, subset_by_index=[0,0])
    # eigvals, eigvecs = eigh(X,driver='ev')
    lambd = eigvecs[:,0]
    # lambd = np.zeros(4*nv)
    # lambd[::4] = 1


    # eigvals, eigvecs = eig(D_Matrix, eigvals_only=False, subset_by_index=[0,0])
    # lambd = V @ first_eigvec
    lambd.shape = (nv, 4)
    lambd /= np.mean(np.linalg.norm(lambd, axis=1))
    lambd.shape = (nv*4, )

    div_e, constraint = new_edges_divergence(trimesh, lambd)
    # Applying constraint to the system
    # TODO make it work for bounded shapes
    c0 = constraint[0]
    c1 = constraint[1]
    for i in [c0[0],c1[0]]:
        for c in [c0]:
            j = c[0]
            div_e[i*4:i*4+4] += L[i*4,j*4]*c[1]
            L[i*4:i*4+4,j*4:j*4+4] = 0
            L[j*4:j*4+4,i*4:i*4+4] = 0


    print(f"{issymmetric(L) = }")
    L = csc_array(L)

    norm_L = norm(L)
    inv_L = inv(L)
    norm_invA = norm(inv_L)
    cond = norm_L*norm_invA
    print(f"L condition number = {cond}")

    new_vertices = spsolve(L, div_e)
    # new_vertices = solve(L, div_e, assume_a='sym')
    residual = np.linalg.norm(L@new_vertices - div_e)
    if residual > 1e-10 :
        print(f"WARNING : {residual =}")

    new_vertices.shape = (nv,4)
    vertices[:,:] = new_vertices[:,1:]

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d
    from matplotlib.colors import LightSource

    def plot_normals(ax,vertices, normals,length=10, color="r"):
        for i in range(normals.shape[0]):
            normalsegm = np.stack((vertices[i],vertices[i,:]+length*normals[i]))
            ax.plot(normalsegm[0,0],normalsegm[0,1],normalsegm[0,2],color+'o')
            ax.plot(normalsegm[:,0],normalsegm[:,1],normalsegm[:,2],color)

    def vertex2face(trimesh, vval):
        assert vval.shape[0] == trimesh.vertices.shape[0]
        return np.mean(vval[trimesh.faces], axis=1)

    trimesh = trimesh.load('sphere.ply')

    vertices = trimesh.vertices
    # nv = vertices.shape[0]
    # triangles = vertices[trimesh.faces]
    # face_center = np.mean(triangles, axis=1)

    dist = np.linalg.norm(vertices-vertices[-1],axis=1)
    rho = 0.01*np.exp(-dist**2/1000)

    transform(trimesh, 0*rho)

    rho_fc = vertex2face(trimesh, rho)
    rho_norm = (rho_fc-rho_fc.min())/np.ptp(rho_fc)
    cm = plt.cm.plasma(rho_norm)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")


    triangles = vertices[trimesh.faces]
    light = LightSource(100,0)
    pc = art3d.Poly3DCollection(triangles, facecolors=cm,
                                shade=True, edgecolors=(1,1,1,0.2),
                                alpha=1, lightsource=light)
    ax.add_collection(pc)
    # plot_normals(ax, face_center, trimesh.face_normals)

    xlim = vertices[:,0].min(), vertices[:,0].max()
    ylim = vertices[:,1].min(), vertices[:,1].max()
    zlim = vertices[:,2].min(), vertices[:,2].max()

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])
    plt.show()
