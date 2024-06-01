#!/usr/bin/env python3
import numpy as np
from numpy.linalg import norm
import trimesh
import networkx as nx
from scipy import sparse
from scipy.linalg import issymmetric
from trimesh_curvature import scalar_curvature, mean_curvature


def mul_quatern(u, v):
    w0, x0, y0, z0 = u
    w1, x1, y1, z1 = v
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def create_dirac_op_sparse(trimesh, rho):
    vertices = trimesh.vertices
    nv = vertices.shape[0]
    assert rho.shape[0] == nv

    ei = np.zeros(4, dtype=np.float64)
    ej = np.zeros(4, dtype=np.float64)
    X_ii = np.zeros(4, dtype=np.float64)
    block_ij = np.zeros((4,4), dtype=np.float64)

    g = nx.from_edgelist(trimesh.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(nv)]

    n_entries = nv
    for i in one_ring:
            n_entries += len(i)
    n_entries *= 16

    one_ordered = [nx.cycle_basis(g.subgraph(i))[0] for i in one_ring]

    print(f"{n_entries =}")
    idx_i = np.zeros(n_entries, dtype=np.int_)
    idx_j = np.zeros(n_entries, dtype=np.int_)
    data = np.zeros(n_entries,dtype=np.float64)

    sp_k = 0
    for i in range(nv):
        # TODO won't work for bounded domain
        if len(one_ordered[i]) >0:
            ring_vert = vertices[one_ordered[i]]
            ring_nv = ring_vert.shape[0]
            vi = vertices[i,:]

            vertex_normal = trimesh.vertex_normals[i]
            ei[1:] = ring_vert[1] - ring_vert[0]
            ej[1:] = ring_vert[0] - vi
            sign = np.dot(vertex_normal,np.cross(ej[1:],ei[1:]))
            sign = int(np.sign(sign))
            ring_vert = ring_vert[::sign,:]
            one_ordered[i] = one_ordered[i][::sign]

            X_ii[:] = 0.0
            for k in range(ring_nv):
                j = one_ordered[i][k]
                ei[1:] = ring_vert[k] - ring_vert[(k-1)%ring_nv]
                ej[1:] = ring_vert[(k-1)%ring_nv] - vi
                #print(ring_vert)
                #print(ei[1:], ej[1:])
                A_x2 = norm(np.cross(ei[1:],ej[1:]))
                X_ij = - mul_quatern(ei, ej)/(2*A_x2) + (rho[i]*ej-rho[j]*ei)/6.0
                X_ij[0] +=  rho[i]*rho[j]*A_x2/18.0

                ei[1:] = ring_vert[(k+1)%ring_nv] - ring_vert[k]
                ej[1:] = vi - ring_vert[(k+1)%ring_nv]
                A_x2 = norm(np.cross(ei[1:],ej[1:]))
                X_ij += -mul_quatern(ei, ej)/(2*A_x2) + (rho[i]*ej-rho[j]*ei)/6.0
                X_ij[0] +=  rho[i]*rho[j]*A_x2/18.0

                block_ij[:,:] = [[X_ij[0], -X_ij[1], -X_ij[2], -X_ij[3]],
                                  [X_ij[1],  X_ij[0], -X_ij[3],  X_ij[2]],
                                  [X_ij[2],  X_ij[3],  X_ij[0], -X_ij[1]],
                                  [X_ij[3], -X_ij[2],  X_ij[1],  X_ij[0]]]


                for l in range(4):
                    for m in range(4):
                        idx_i[sp_k] = i*4 + l
                        idx_j[sp_k] = j*4 + m
                        data[sp_k] = block_ij[l,m]
                        sp_k += 1

                X_ii -= mul_quatern(ei, ei)/(2*A_x2)
                X_ii[0] +=  rho[i]*rho[i]*A_x2/18.0

            block_ij[:,:] = [[X_ii[0], -X_ii[1], -X_ii[2], -X_ii[3]],
                                [X_ii[1],  X_ii[0], -X_ii[3],  X_ii[2]],
                                [X_ii[2],  X_ii[3],  X_ii[0], -X_ii[1]],
                                [X_ii[3], -X_ii[2],  X_ii[1],  X_ii[0]]]


            for l in range(4):
                for m in range(4):
                    idx_i[sp_k] = i*4 + l
                    idx_j[sp_k] = i*4 + m
                    data[sp_k] = block_ij[l,m]
                    sp_k += 1

    print(f"{sp_k =}")
    return idx_i, idx_j, data


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


    i_const = 0
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
            for k in range(ring_nv):
                e1 = -edges_vect[(k-1)%ring_nv]
                o1 =  edges_vect[k]+e1
                cos1 = np.dot(e1,o1)
                sin1 = norm(np.cross(o1,e1))
                cot1 = cos1/sin1

                e2 = -edges_vect[(k+1)%ring_nv]
                o2 =  edges_vect[k]+e2
                cos2 = np.dot(e2,o2)
                sin2 = norm(np.cross(e2,o2))

                cot2 = cos2/sin2

                j = one_ordered[i][0][k]
                eij[1:] = edges_vect[k]
                e_new  = 1/3 * mul_quatern(lambdc[i*4:i*4+4], mul_quatern(eij, lambd[i*4:i*4+4]))
                e_new += 1/6 * mul_quatern(lambdc[i*4:i*4+4], mul_quatern(eij, lambd[j*4:j*4+4]))
                e_new += 1/6 * mul_quatern(lambdc[j*4:j*4+4], mul_quatern(eij, lambd[i*4:i*4+4]))
                e_new += 1/3 * mul_quatern(lambdc[j*4:j*4+4], mul_quatern(eij, lambd[j*4:j*4+4]))
                # print(norm(e_new-eij))

                div[4*i+1:4*i+4] += e_new[1:]*(cot2+cot1)
                i_const = i

    constraint[0][0] = i_const
    constraint[0][1][1:] = trimesh.vertices[i_const,:]
    constraint[1][0] = j
    constraint[1][1][1:] =  trimesh.vertices[i_const,:]+e_new[1:]

    return div, constraint


def quaternionic_laplacian_matrix(trimesh):
    nv = trimesh.vertices.shape[0]
    g = nx.from_edgelist(trimesh.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(nv)]
    one_ordered = [nx.cycle_basis(g.subgraph(i)) for i in one_ring]
    L = np.zeros((nv*4,nv*4))

    n_entries = nv
    for i in one_ring:
            n_entries += len(i)
    n_entries *= 16
    idx_i = np.zeros(n_entries, dtype=np.int_)
    idx_j = np.zeros(n_entries, dtype=np.int_)
    data = np.zeros(n_entries,dtype=np.float64)

    cot = 0.0
    diag = 0.0
    sp_k = 0
    for i in range(nv):
        if len(one_ordered[i][0]) >0:
            ring_vert = trimesh.vertices[one_ordered[i][0]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert-trimesh.vertices[i,:]
            vertex_normal = trimesh.vertex_normals[i]
            sign = np.dot(vertex_normal,np.cross(edges_vect[0],edges_vect[1]))
            sign = int(np.sign(sign))
            edges_vect = edges_vect[::sign,:]
            one_ordered[i][0] = one_ordered[i][0][::sign]
            ring_vert = ring_vert[::sign,:]

            ring_nv = ring_vert.shape[0]
            # iterating over each of the edges adjacent to the vertex i
            diag = 0.0
            for k in range(ring_nv):
                e1 = -edges_vect[(k-1)%ring_nv]
                o1 =  edges_vect[k]+e1
                cos1 = np.dot(e1,o1)
                sin1 = norm(np.cross(o1,e1))
                cot = cos1/sin1

                e2 = -edges_vect[(k+1)%ring_nv]
                o2 =  edges_vect[k]+e2

                cos2 = np.dot(e2,o2)
                sin2 = norm(np.cross(e2,o2))

                cot += cos2/sin2

                j = one_ordered[i][0][k]
                diag -= cot
                for l in range(4):
                    idx_i[sp_k] = i * 4 + l
                    idx_j[sp_k] = j * 4 + l
                    data[sp_k] = cot
                    sp_k += 1

            for l in range(4):
                idx_i[sp_k] = i * 4 + l
                idx_j[sp_k] = i * 4 + l
                data[sp_k] = diag
                sp_k += 1

    return idx_i, idx_j, data

def eigensolve(M, v):
    nv = v.shape[0]
    v[:] = 0
    v[::4] = 1.0
    # eigval, eigvect= sparse.linalg.eigsh(M, k=1, sigma=0, which='LM', tol=0)
    # v = eigvect[:, 0]
    for i in range(5):
        v[:] = sparse.linalg.spsolve(M, v)
        v.shape = (nv//4,4)
        v /= np.sqrt(np.sum(v**2))
        v.shape = (nv, )

    v_res = M@v
    print(norm(v_res)/norm(v))
    v_res.shape = (nv//4,4 )
    v_res /= np.sqrt(np.sum(v_res**2))
    v_res.shape = (nv, )
    print("eigen vector residual inf norm :",norm(v_res - v , ord=np.inf))
    v.shape = (nv//4,4)
    v /= np.mean(np.sqrt(np.sum(v**2,axis=1)))
    v.shape = (nv, )


def symetric_delete(i_del, idx_i, idx_j, data, n):
    idx_table = np.empty(n, np.int_)
    idx_sp = 0
    for k in range(n):
        if k not in i_del:
            idx_table[k] = idx_sp
            idx_sp +=1

    idx_sp = 0
    for k in range(data.shape[0]):
        if (idx_i[k] not in i_del) and (idx_j[k] not in i_del):
            idx_i[idx_sp] = idx_table[idx_i[k]]
            idx_j[idx_sp] = idx_table[idx_j[k]]
            data[idx_sp] = data[k]
            idx_sp += 1

    idx_i = idx_i[:idx_sp]
    idx_j = idx_j[:idx_sp]
    data = data[:idx_sp]
    return idx_i, idx_j, data

# def symetric_delete2(i_del, L):
#     L_wp = np.delete(L,i_del, axis=0)
#     return np.delete(L_wp, i_del, axis=1)


def transform(trimesh, rho):

    vertices = trimesh.vertices
    nv = vertices.shape[0]

    d_i, d_j, d_data = create_dirac_op_sparse(trimesh, rho)
    X = sparse.csc_array((d_data,(d_i,d_j)))
    X = (X + X.T)/2
    print(f"{np.abs(X-X.T).max() = }")



    lambd = np.zeros(4*nv)
    eigensolve(X, lambd)

    div_e, constraint = new_edges_divergence(trimesh, lambd)

    idx_i, idx_j, data = quaternionic_laplacian_matrix(trimesh)

    # csc for row slicing
    L = sparse.csc_array((data, (idx_i, idx_j)),shape=(nv*4,nv*4))

    # Applying constraint to the system
    # TODO make it work for bounded shapes
    for c in constraint:
        i = c[0]
        div_e[:] -= L[:,i*4:i*4+4] @ c[1].T

    i_sorted = np.argsort([constraint[0][0],constraint[1][0]])
    i1 = constraint[i_sorted[0]][0]
    i2 = constraint[i_sorted[1]][0]

    i_del = [ic*4+i for ic in [i1, i2] for i in range(4)]

    idx_i, idx_j, data = symetric_delete(i_del, idx_i, idx_j, data, nv*4)

    # csr
    L = sparse.csc_array((data, (idx_i, idx_j)),shape=((nv-2)*4,(nv-2)*4))
    # Ls = (L + L.T)/2
    # print(f"{issymmetric(L.todense()) = }")

    div_wp = np.delete(div_e, i_del, axis=0)


    # norm_L = norm(L)
    # inv_L = inv(L)
    # norm_invA = norm(inv_L)
    # cond = norm_L*norm_invA
    # print(f"L condition number = {cond}")

    new_vertices_wp = sparse.linalg.spsolve(L, div_wp)
    # new_vertices = solve(L, div_e, assume_a='sym')
    residual = norm(L@new_vertices_wp - div_wp)
    if residual > 1e-10 :
        print(f"WARNING : {residual =}")

    new_vertices_wp.shape = (nv-2,4)
    new_vertices = np.insert(new_vertices_wp, i1, constraint[i_sorted[0]][1], axis=0)
    new_vertices = np.insert(new_vertices, i2, constraint[i_sorted[1]][1], axis=0)
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

    # trimesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0],[0,0,1]],
    #                     faces=[[0, 1, 3],[1,2,3],[2,0,3],[0,2,1]])
    trimesh = trimesh.load('meshes/sphere.ply')
    print("number of vertices", trimesh.vertices.shape[0])

    kN = mean_curvature(trimesh)
    k = scalar_curvature(trimesh, kN)
    mk = np.mean(k)
    print("mean curvature =", mk)

    vertices = trimesh.vertices
    nv = vertices.shape[0]


    R = 50
    pts = []
    pts += [[np.pi/4, 2*np.pi/6]]
    pts += [[-3*np.pi/4, 0]]
    pts += [[-np.pi/4, -np.pi/6]]
    ampl = np.array([-15, -10, 18])*abs(mk)
    rad = [100.0, 150.0, 100.0]

    rho = np.zeros(nv,dtype=np.float64)
    for i, pt in enumerate(pts):
        pt_cart = np.array([R*np.cos(pt[1])*np.cos(pt[0]),
                    R*np.cos(pt[1])*np.sin(pt[0]),
                    R*np.sin(pt[1])])
        dist = norm(vertices-pt_cart,axis=1)
        rho += ampl[i]*np.exp(-dist**2/rad[i])



    rho_fc = vertex2face(trimesh, rho)
    rho_norm = (rho_fc-rho_fc.min())/np.ptp(rho_fc)
    cm = plt.cm.plasma(rho_norm)

    transform(trimesh, rho)

    fig, ax = plt.subplots(1,1,figsize=(8,4),subplot_kw=dict(projection='3d'))


    triangles = vertices[trimesh.faces]
    light = LightSource(90,100)
    pc = art3d.Poly3DCollection(triangles, facecolors=cm,
                                shade=True, edgecolors=(1,1,1,0.2),
                                alpha=1, lightsource=light)
    ax.add_collection(pc)
    # plot_normals(ax, face_center, trimesh.face_normals)
    xlim = (vertices[:,0].min(), vertices[:,0].max())
    ylim = (vertices[:,1].min(), vertices[:,1].max())
    zlim = (vertices[:,2].min(), vertices[:,2].max())

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])

    fig.tight_layout(pad=0)
    plt.show()
