#%%
from mpi4py import MPI
from dolfinx import fem
from dolfinx.mesh import create_unit_cube, locate_entities_boundary
from dolfinx.fem import Function, dirichletbc, Expression, locate_dofs_topological, form, petsc
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
from ufl import SpatialCoordinate, as_vector, grad, pi, curl, cos, TrialFunction, TestFunction, inner, dx
from basix.ufl import element
from petsc4py import PETSc
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from utils import boundary_marker, L2_norm, par_print, error_L2, monitor
from problems import quadratic, sinusodial
from ufl import extract_blocks

comm = MPI.COMM_WORLD

case = 3
degree = 1

n = 8

domain = create_unit_cube(MPI.COMM_WORLD, n, n, n)
gdim = domain.geometry.dim
facet_dim = gdim - 1 #Topological dimension 

DG = fem.functionspace(domain, ("DG", 0))
alpha = Function(DG)
alpha.interpolate(lambda x: np.where(x[0] <= 0.5, 1.0, 1.0))



nedelec_elem = element("N1curl", domain.basix_cell(), degree)
A_space = fem.functionspace(domain, nedelec_elem)

lagrange_elem = element("Lagrange", domain.basix_cell(), degree)
V1 = fem.functionspace(domain, lagrange_elem)

# Mixed function space

from ufl import MixedFunctionSpace, extract_blocks
import ufl

W = MixedFunctionSpace(A_space, V1)

A, phi = ufl.TrialFunctions(W)
v, psi = ufl.TestFunctions(W)

x = SpatialCoordinate(domain)
u_e = sinusodial(x)
f = curl(alpha * curl(u_e))

a_all = inner(alpha * curl(A), curl(v)) * dx + inner(grad(phi), v) * dx + inner(A, grad(psi)) * dx
L_all = inner(f,v) * dx

a = form(extract_blocks(a_all))
L = form(extract_blocks(L_all))



facets = locate_entities_boundary(domain, dim=(domain.topology.dim - 1), marker=boundary_marker)
dofs_V = locate_dofs_topological(V=A_space, entity_dim=facet_dim, entities=facets)
dofs_V1 = locate_dofs_topological(V=V1, entity_dim=facet_dim, entities=facets)

u_bc_expr = Expression(u_e, A_space.element.interpolation_points)
# u_bc_expr_V1 = Expression(u_e, V1.element.interpolation_points)

u_bc = Function(A_space)
u_bc.interpolate(u_bc_expr)
bc = dirichletbc(u_bc, dofs_V)

u_bc_V1 = Function(V1)
bc_V1 = dirichletbc(u_bc_V1, dofs_V1)


# Solver steps

A_mat = assemble_matrix(a, bcs = [bc])
A_mat.assemble()


from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.fem import bcs_by_block, extract_function_spaces

b = assemble_vector(L, kind=PETSc.Vec.Type.MPI)
bcs1 = bcs_by_block(extract_function_spaces(a, 1), bc)
apply_lifting(b, a, bcs=bcs1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bcs0 = bcs_by_block(extract_function_spaces(L), bc)
set_bc(b, bcs0)


uh = fem.Function(A_space)


iterative_opts = {"ksp_atol": 1e-8, 
           "ksp_rtol": 1e-8,
           "ksp_type": "gmres"}

ams_opts = {
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_type": "gmres",
    "pc_hypre_ams_cycle_type": 7,
    "pc_hypre_ams_tol": 1,
    "pc_hypre_ams_max_iter": 1,
    "pc_hypre_ams_amg_beta_theta": 0.25,
    "pc_hypre_ams_print_level": 1,
    "pc_hypre_ams_amg_alpha_options": "10,1,3",
    "pc_hypre_ams_amg_beta_options": "10,1,3",
}

# Iterative Solver with Direct LU preconditioner

if case == 1:
    print("Case 1: Direct Solver")

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A_mat)
    ksp.setType('preonly')
    
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")

    opts = PETSc.Options()
    opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
    opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
    opts["ksp_error_if_not_converged"] = 1
    ksp.setFromOptions()
   
    ksp.setUp()
    pc.setUp()

# Iterative Solver with AMS Preconditioner

if case == 2:
    print("Case 2: Iterative Solver with AMS Preconditioner")

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A_mat)
    ksp.setOptionsPrefix(f"ksp_{id(ksp)}")

    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for option, value in ams_opts.items():
        opts[option] = value
    opts.prefixPop()
    
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("ams")

    # Build discrete gradient
    V_CG = fem.functionspace(domain, ("CG", degree))._cpp_object
    G = discrete_gradient(V_CG, A_space._cpp_object)
    G.assemble()
    pc.setHYPREDiscreteGradient(G)

    if degree == 1:
            cvec_0 = Function(A_space)
            cvec_0.interpolate(lambda x: np.vstack((np.ones_like(x[0]),
                                                    np.zeros_like(x[0]),
                                                    np.zeros_like(x[0]))))
            cvec_1 = Function(A_space)
            cvec_1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                                    np.ones_like(x[0]),
                                                    np.zeros_like(x[0]))))
            cvec_2 = Function(A_space)
            cvec_2.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                                    np.zeros_like(x[0]),
                                                    np.ones_like(x[0]))))
            pc.setHYPRESetEdgeConstantVectors(cvec_0.x.petsc_vec,
                                                    cvec_1.x.petsc_vec,
                                                    cvec_2.x.petsc_vec)
    else:   
            Vec_CG = fem.functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
            Pi = interpolation_matrix(Vec_CG._cpp_object, A_space._cpp_object)
            Pi.assemble()

            # Attach discrete gradient to preconditioner
            pc.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)

    pc.setHYPRESetBetaPoissonMatrix(None)

    ksp.setFromOptions()
    ksp.setUp()
    pc.setUp()

if case == 3:
    
    print("Case 3: Iterative Solver with AMG")

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A_mat)
    ksp.setOptionsPrefix(f"ksp_{id(ksp)}")
    ksp.setType("gmres")
    
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for option, value in iterative_opts.items():
        opts[option] = value
    opts.prefixPop()

    pc = ksp.getPC()
    pc.setType("gamg")
    # pc.setFactorSolverType("mumps")
    
    ksp.setFromOptions()
    ksp.setUp()
    pc.setUp()

#%%
ksp.setMonitor(monitor)
ksp.solve(b, uh.x.petsc_vec)

res = A_mat * uh.x.petsc_vec - b
# print("Residual norm: ", res.norm())

iterations = ksp.getIterationNumber()
par_print(comm, f"Number of iterations: {iterations}")

reason = ksp.getConvergedReason()
par_print(comm, f"Convergence reason: {reason}")

error_B = L2_norm(curl(uh - u_e))

error_uh = L2_norm(uh - u_e)

par_print(comm, f"n is {n}")
par_print(comm, f"Error in B: {error_B}")
par_print(comm, f"Error in uh: {error_uh}")

# %%
