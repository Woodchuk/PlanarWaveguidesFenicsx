import sys
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from slepc4py import SLEPc
from petsc4py import PETSc
from mpi4py import MPI as nMPI
import meshio
import ufl
import basix.ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmshio
from dolfinx import fem
from dolfinx import io
import gmsh

comm = nMPI.COMM_WORLD
mpiRank = comm.rank
gmshRank = 0

def curl_2d(u, v, beta):
    ax = v.dx(1) + 1j * beta * u[1]
    ay = -1j * beta * u[0] - v.dx(0)
    return ufl.as_vector ((ax, ay))
    
def perm_vec(u):
    return ufl.as_vector((-u[1], u[0]))
    

def Beta(k0, Symm):
    h = 0.508
    w = 1.05
    t = 0.035
    s = 0.2
    a = 10.0
    b = 5.0
    Dk = 3.66
    eta0 = 377.0
    
    eps = 1.0e-3
    ls = 0.01
    lm = 0.25
    
    if mpiRank == gmshRank:
        print("K0 = {0:<f}".format(k0))
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add("Microstrip WG")
        gmsh.model.setCurrent("Microstrip WG")

        gmsh.model.occ.addRectangle(0, 0, 0, a/2, h, tag=1)
        gmsh.model.occ.addRectangle(0, h, 0, a/2, b-h, tag=2)
        gmsh.model.occ.addRectangle(s/2, h, 0, w, t, tag=3)
        gmsh.model.occ.cut([(2,2)],[(2,3)], 4, removeObject=True, removeTool=True)
        
        gmsh.model.occ.synchronize()
        ov, ovv = gmsh.model.occ.fragment([(2,4)],[(2,1)], -1)
        gmsh.model.occ.synchronize()
        pt = gmsh.model.getEntities(0)
        print(pt)
        gmsh.model.mesh.setSize(pt, lm)
        ov = gmsh.model.getEntitiesInBoundingBox(s/2-eps, h-eps, -eps, s/2+w+eps, h+t+eps, eps)
        print(ov)
        gmsh.model.mesh.setSize(ov, ls)
  
        PEC = []
        PMC = []
        Strip1 = []
        bb = gmsh.model.getEntities(dim=2)
        pt = gmsh.model.getBoundary(bb, combined=True, oriented=False, recursive=False) # All BCs    
        for bnd in pt:
            CoM = gmsh.model.occ.getCenterOfMass(bnd[0], bnd[1])
            if(np.abs(CoM[1] - h) < eps) or (np.abs(CoM[1] -h -t) < eps) or (np.abs(CoM[1] - h- t/2) < eps):
                Strip1.append(bnd[1])
            elif (CoM[0] < eps) and (Symm == 1):
                PMC.append(bnd[1])
            else:
                PEC.append(bnd[1])
 
                
        gmsh.model.addPhysicalGroup(1, PEC, 1)
        gmsh.model.addPhysicalGroup(1, PMC, 2)
        gmsh.model.addPhysicalGroup(1, Strip1, 3)
        gmsh.model.setPhysicalName(1, 1, "PEC")
        gmsh.model.setPhysicalName(1, 2, "PMC")
        gmsh.model.setPhysicalName(1, 3, "Strip1")
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.model.setPhysicalName(2, 1, "Substrate")
        gmsh.model.addPhysicalGroup(2, [4], 2)
        gmsh.model.setPhysicalName(2, 2, "Air")

        gmsh.option.setNumber('Mesh.MeshSizeMin', ls)
        gmsh.option.setNumber('Mesh.MeshSizeMax', lm)        
        gmsh.model.mesh.generate(2)
        
#        gmsh.fltk.run()
    mesh, ct, fm = gmshio.model_to_mesh(gmsh.model, comm, gmshRank, gdim=2)
    if mpiRank == gmshRank:
        gmsh.finalize()
      
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    degree = 4
    RTCE = basix.ufl.element('Nedelec 1st kind H(curl)', mesh.basix_cell(), degree=degree)
    Q = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=degree)
    V = fem.functionspace(mesh, basix.ufl.mixed_element([RTCE, Q]))

    et, ez = ufl.TrialFunctions(V)
    vt, vz = ufl.TestFunctions(V)

    Q = fem.functionspace(mesh,("DG", 0))
    Dk_f = fem.Function(Q)
    Subst = ct.find(1)
    Air = ct.find(2)
    Dk_f.x.array[Subst] = np.full_like(Subst, Dk, dtype=PETSc.ScalarType)
    Dk_f.x.array[Air] = np.full_like(Air, 1.0, dtype=PETSc.ScalarType)
    
    with io.XDMFFile(mesh.comm, "Dk.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(Dk_f)
        
    n = ufl.FacetNormal(mesh)
    tau = ufl.as_vector((-n[1], n[0]))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=fm)

        
    a_tt = (ufl.inner(ufl.curl(et), ufl.curl(vt)) - k0
        ** 2 * Dk_f * ufl.inner(et, vt)) * ufl.dx
    b_tt = ufl.inner(et, vt) * ufl.dx
    b_tz = ufl.inner(et, ufl.grad(vz)) * ufl.dx
    b_zt = ufl.inner(ufl.grad(ez), vt) * ufl.dx
    b_zz = (ufl.inner(ufl.grad(ez), ufl.grad(vz)) - k0
        ** 2 * Dk_f * ufl.inner(ez, vz)) * ufl.dx

    a = fem.form(a_tt)
    b = fem.form(b_tt + b_tz + b_zt + b_zz)
        
    bc_facets1 = fm.find(1)
    bc_facets2 = fm.find(3)
    bc_dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, bc_facets1)
    bc_dofs1 = np.append(bc_dofs, fem.locate_dofs_topological(V, mesh.topology.dim - 1, bc_facets2))
    u_bc = fem.Function(V)
    with u_bc.x.petsc_vec.localForm() as loc:
        loc.set(0)
    bc = fem.dirichletbc(u_bc, bc_dofs1)
    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    B = assemble_matrix(b, bcs=[bc])
    B.assemble()

    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    print(f"Number of dofs (owned) by rank {mpiRank}: {num_dofs_local}")
    if mpiRank == gmshRank:
       print(f"Number of dofs global: {num_dofs_global}")
    eigs = SLEPc.EPS().create(mesh.comm)
    eigs.setOperators(A, B)
    eigs.setProblemType(SLEPc.EPS.ProblemType.GNHEP) # General non hermitian
    eigs.setTolerances(1.0e-9)
    eigs.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    st = eigs.getST() # Invoke spectral transformation
    st.setType(SLEPc.ST.Type.SINVERT)
    eigs.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL) # target real part of eigenvalue
    eigs.setTarget(-k0*k0*Dk/2)
    eigs.setDimensions(nev=12) # Number of eigenvalues
    eigs.solve()
    eigs.view()
    eigs.errorView()
    evs = eigs.getConverged()
    eh = fem.Function(V)
    if mpiRank==gmshRank:
       print( "Number of converged eigenpairs %d" % evs )
    vals = []
    if evs > 0:
        for i in range (evs):
            error = 0.0
            l = eigs.getEigenvalue(i)
            vals.append((i,np.sqrt(-l)))
            if(l.real < 1.0e-3):
                 error = eigs.getErrorEstimate(i)
            if mpiRank == gmshRank:
                if l.real < 1.0e-3:
                    print("i= {3} kz = {0}, Zm = {1}, error = {2}".format(np.sqrt(-l), 377 * k0 / np.sqrt(-l), error, i))
        vals.sort(key= lambda x: x[1].real)
#        Vw = basix.ufl.element('DQ', mesh.basix_cell(), 0, shape=(mesh.geometry.dim, ))
        W = fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim, )))
        Et_dg = fem.Function(W)
        
        for i, kz in vals:
            eigs.getEigenpair(i, eh.x.petsc_vec)
            eh.x.scatter_forward()
            eth, ezh = eh.split()
            eth = eh.sub(0).collapse()
            ezh = eh.sub(1).collapse()
            
            if kz > eps:
                eth.x.array[:] = eth.x.array[:] / kz
                ezh.x.array[:] = ezh.x.array[:] * 1j
                Et_dg.interpolate(eth)
                with io.XDMFFile(mesh.comm, "Efield_{0}_{1}.xdmf".format(i, Symm), "w") as xx:
                    xx.write_mesh(mesh)
                    xx.write_function(Et_dg)

                H_expr = fem.Expression(curl_2d(eth, ezh, kz), W.element.interpolation_points())
                Et_dg.interpolate(H_expr)
                with io.XDMFFile(mesh.comm, "Hfield_{0}_{1}.xdmf".format(i, Symm), "w") as xx:
                    xx.write_mesh(mesh)
                    xx.write_function(Et_dg)
                    
                Curr1 = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(tau, curl_2d(eth, ezh, kz)) *  ds(3))), op = nMPI.SUM) / (1j * k0 * eta0)
                Pm = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(perm_vec(eth), curl_2d(eth, ezh, kz)) * ufl.dx)), op=nMPI.SUM) / (1j * k0 * eta0)

                print("Current = {0}, Power = {1}, Char. Impedance = {2}".format(Curr1, Pm, Pm / np.abs(Curr1)**2 / 2.0))
    return (kz, np.abs(Curr1), 2.0 * np.real(Pm))

k0 = 0.05    
Im = np.zeros((11, 2, 2), dtype=np.complex128)
Pm = np.zeros((11, 2, 2), dtype=np.complex128)
Dk_eff = np.zeros((11, 2), dtype=np.float64)
Z0 = np.zeros((11, 2, 2), dtype=np.float64)
k0 = np.zeros(11, dtype=np.float64)
for n in range(11):
    k0[n] = 0.05+0.25 * n / 10.0
    k1, I1, P1 = Beta(k0[n], 1)
    k2, I2, P2 = Beta(k0[n], 0)
    Im[n] = np.array([[I1, I1],[-I2, I2]])
    Pm[n] = np.array([[P1, 0], [0, P2]])
    Dk_eff[n] = np.array([(k1/k0[n])**2, (k2/k0[n])**2])
    Z0[n] = np.real(np.matmul(inv(Im[n]), np.matmul(Pm[n], np.transpose(np.conjugate(inv(Im[n]))))    )) 
    
    if mpiRank == gmshRank:
        print(Im[n])
        print(Pm[n])
        print(Dk_eff[n])
        print(Z0[n])
        
if mpiRank == gmshRank:
    fig, ax = plt.subplots(2, 1, facecolor='sienna')
    DD = np.array(Dk_eff[:, 0])
    ps = ax[0].plot(k0, DD, 'r-')
    DD = np.array(Dk_eff[:, 1])
    pq = ax[0].plot(k0, DD, 'g-')
    pt = mp.Patch(color='red', label='Dk mode 1')
    pu = mp.Patch(color='green', label='Dk mode 2')
    l1 = ax[0].legend(handles=[pt, pu], loc='center right', shadow=True)
    ax[0].grid = True
    ax[0].set_ylabel('Eff. Dk')
    ax[0].set_xlabel('k0')

    ps = ax[1].plot(k0, Z0[:, 0, 0], 'r-')
    pq = ax[1].plot(k0, Z0[:, 0, 1], 'g-')
    pu = ax[1].plot(k0, Z0[:, 1, 1], 'b-')
    pt = mp.Patch(color='red', label='Z11')
    pv = mp.Patch(color='green', label='Z12, Z21')
    pw = mp.Patch(color='blue', label='Z22')
    l1 = ax[1].legend(handles=[pt, pv, pw], loc='center right', shadow=True)
    ax[1].grid = True
    ax[1].set_ylabel('Char. Impedance')
    ax[1].set_xlabel('k0')
    plt.show()

sys.exit(0)

        
