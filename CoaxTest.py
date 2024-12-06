import sys
import numpy as np
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
    

def Beta(k0):
    a = 1.0
    b = 3.441
    Dk = 2.2
    eta0 = 377.0
    
    eps = 1.0e-3
    ls = 0.01
    lm = 0.1
    
    if mpiRank == gmshRank:
        print("K0 = {0:<f}".format(k0))
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add("Coax")
        gmsh.model.setCurrent("Coax")

        gmsh.model.occ.addDisk(0, 0, 0, b/2, b/2, tag=1)
        gmsh.model.occ.addDisk(0, 0, 0, a/2, a/2, tag=2)
        gmsh.model.occ.cut([(2,1)], [(2,2)], tag=3, removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        PEC = []
        Cntr = []
        bb = gmsh.model.getEntities(dim=2)
        pt = gmsh.model.getBoundary(bb, combined=True, oriented=False, recursive=False) # All BCs    
        for bnd in pt:
            Mass = gmsh.model.occ.getMass(bnd[0], bnd[1])
            if Mass < np.pi * (b + a) / 2.0:
                Cntr.append(bnd[1])
            else:    
                PEC.append(bnd[1])
 
                
        gmsh.model.addPhysicalGroup(1, PEC, 1)
        gmsh.model.setPhysicalName(1, 1, "PEC")
        gmsh.model.addPhysicalGroup(1, Cntr, 2)
        gmsh.model.setPhysicalName(1, 2, "Center Conductor")        
        gmsh.model.addPhysicalGroup(2, [3], 1)
        gmsh.model.setPhysicalName(2, 1, "Dielectric")
        
        pt = gmsh.model.getEntities(0)
        print(pt)
        gmsh.model.mesh.setSize(pt, lm)
        gmsh.option.setNumber('Mesh.MeshSizeMin', ls)
        gmsh.option.setNumber('Mesh.MeshSizeMax', lm)        
        gmsh.model.mesh.generate(2)
        
        gmsh.fltk.run()
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
    
    n = ufl.FacetNormal(mesh)
    tau = ufl.as_vector((-n[1], n[0]))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=fm)

        
    a_tt = (ufl.inner(ufl.curl(et), ufl.curl(vt)) - k0
        ** 2 * Dk * ufl.inner(et, vt)) * ufl.dx
    b_tt = ufl.inner(et, vt) * ufl.dx
    b_tz = ufl.inner(et, ufl.grad(vz)) * ufl.dx
    b_zt = ufl.inner(ufl.grad(ez), vt) * ufl.dx
    b_zz = (ufl.inner(ufl.grad(ez), ufl.grad(vz)) - k0
        ** 2 * Dk * ufl.inner(ez, vz)) * ufl.dx

    a = fem.form(a_tt)
    b = fem.form(b_tt + b_tz + b_zt + b_zz)
        
    bc_facets1 = fm.find(1)
    bc_facets2 = fm.find(2)
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
        W = fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim, )))
        Et_dg = fem.Function(W)
# plot fields        
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
# E-field
                with io.XDMFFile(mesh.comm, "Efield_{0}.xdmf".format(i), "w") as xx:
                    xx.write_mesh(mesh)
                    xx.write_function(Et_dg)
# H-field
                H_expr = fem.Expression(curl_2d(eth, ezh, kz), W.element.interpolation_points())
                Et_dg.interpolate(H_expr)
                with io.XDMFFile(mesh.comm, "Hfield_{0}.xdmf".format(i), "w") as xx:
                    xx.write_mesh(mesh)
                    xx.write_function(Et_dg)

# Integrate magnetic field along center conductor  to extract current (Ampere's Law)                  
                Curr1 = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(tau, curl_2d(eth, ezh, kz)) *  ds(2))), op = nMPI.SUM) / (1j * k0 * eta0)
# Integrate Poynting vector over area to get mode power.
                Pm = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(perm_vec(eth), curl_2d(eth, ezh, kz)) * ufl.dx)), op=nMPI.SUM) / (1j * k0 * eta0)

                print("Current = {0}, Power = {1}, Char. Impedance = {2}".format(Curr1, Pm, Pm / np.abs(Curr1)**2))
    return 0.0

 
Beta(0.1)


        
