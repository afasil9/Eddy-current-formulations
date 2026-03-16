# %%
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import (
    functionspace,
    bcs_by_block,
    extract_function_spaces,
)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from ufl import (
    grad,
    variable,
    curl,
    Measure,
)
from dolfinx.fem import Function, dirichletbc, form
import numpy as np
from basix.ufl import element
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from utils import par_print
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import GhostMode
from utils import my_monitor, interpolate_by_tags, boundary_marker_copper, L2_norm
from dolfinx import default_scalar_type
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological


comm = MPI.COMM_WORLD

with XDMFFile(comm, "copper_rod.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(ghost_mode=GhostMode.none)
    ct = xdmf.read_meshtags(domain, name="ct")
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, 0)
    ft = xdmf.read_meshtags(domain, name="ft")
    material_tags = np.unique(ct.values)
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim) 

boundary_tags = {
    "cube_boundary": 1,
    "upper_surface": 2,
    "side_surface": 3,
    "bottom_surface": 4
}

vol_ids = {
    "copper": 1,
    "air": 2 
    }


ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 500  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

degree = 1

const = fem.functionspace(domain, ("DG", 0)) #Piecewise constant function space

sigma = fem.Function(const)
nu = fem.Function(const)

sigma_air = fem.Constant(domain, default_scalar_type(0.0))
sigma_copper = fem.Constant(domain, default_scalar_type(5.96e7))
nu_value = fem.Constant(domain, default_scalar_type(1e6))

sigma_values = {
    vol_ids["air"]: sigma_air,
    vol_ids["copper"]: sigma_copper,
}

nu_values = {
    vol_ids["air"]: nu_value,
    vol_ids["copper"]: nu_value
}

interpolate_by_tags(sigma, sigma_values, ct)
interpolate_by_tags(nu, nu_values, ct)


nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = functionspace(domain, nedelec_elem)

lagrange_elem = element("Lagrange", domain.basix_cell(), degree)
V1 = functionspace(domain, lagrange_elem)

W = fem.functionspace(domain, ("Lagrange", degree))
interior_nodes_array = fem.Function(W)

interior_nodes_array.x.array[:] = 1.0
interior_nodes_array.x.scatter_forward()

dofmap = W.dofmap
num_dofs_per_cell = dofmap.dof_layout.num_dofs
cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

tagged_cells = ct.find(vol_ids["copper"])

tagged_cell_dofs = cell_dofs[tagged_cells].flatten()
unique_dofs = np.unique(tagged_cell_dofs)

interior_nodes_array.x.array[unique_dofs] = 0.0
interior_nodes_array.x.scatter_forward()


gdim = domain.geometry.dim
facet_dim = gdim - 1


frequency = 50.0
omega = 2.0 * np.pi * frequency
V_in = 10.0


outer_boundary_facets = ft.find(boundary_tags["cube_boundary"])

boundary_tags_total = (1,2,3,4)
boundary_facets_V = np.concatenate([ft.find(tag) for tag in boundary_tags_total])


bdofs0 = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=outer_boundary_facets)
u_bc_V = fem.Function(V)
u_bc_V.x.array[:] = 0
bc_ex = fem.dirichletbc(u_bc_V, bdofs0)

bdofs1 = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=outer_boundary_facets)
u_bc_V1 = fem.Function(V1)
u_bc_V1.x.array[:] = 0.0
bc_ex1 = fem.dirichletbc(u_bc_V1, bdofs1)

upper_facets = ft.find(boundary_tags["upper_surface"])
bdofs2 = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=upper_facets)
high_func = fem.Function(V1)

high_expr = fem.Expression(
    V_in * ufl.sin(omega * t), V1.element.interpolation_points
)
high_func.interpolate(high_expr)
# high_func.x.array[:] = 10.0

bc_ex2 = fem.dirichletbc(high_func, bdofs2)

lower_facets = ft.find(boundary_tags["bottom_surface"])
bdofs3 = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=lower_facets)
low_func = fem.Function(V1)
low_func.x.array[:] = 0.0
bc_ex3 = fem.dirichletbc(low_func, bdofs3)

bc = [bc_ex, bc_ex1,bc_ex2, bc_ex3]

u_n = fem.Function(V)
u_n1 = fem.Function(V1)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

dx = Measure("dx", domain=domain, subdomain_data=ct)


a00 = dt * ufl.inner(nu * curl(u), curl(v)) * dx + ufl.inner(sigma * u, v) * dx

a01 = dt * ufl.inner(sigma * grad(u1), v) * dx(vol_ids["copper"])
a10 = ufl.inner(sigma * grad(v1), u) * dx(vol_ids["copper"])

a11 = dt * ufl.inner(sigma * ufl.grad(u1), ufl.grad(v1)) * dx(vol_ids["copper"])

L0 = ufl.inner(sigma * u_n, v) * dx(vol_ids["copper"])
L1 = ufl.inner(grad(v1), sigma * u_n) * dx

a = form([[a00, a01], [a10, a11]])

par_print(comm, "Assembling system matrix...")

A_mat = assemble_matrix(a, bcs=bc)
A_mat.assemble()
par_print(comm, f"A matrix norm is {A_mat.norm()}")


L = form([L0, L1])

b = assemble_vector(L, kind=PETSc.Vec.Type.MPI)
bcs1 = bcs_by_block(extract_function_spaces(a, 1), bc)
apply_lifting(b, a, bcs=bcs1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bcs0 = bcs_by_block(extract_function_spaces(L), bc)
set_bc(b, bcs0)


a_p = form([[a00, None], [None, a11]])

P = assemble_matrix(a_p, bcs=bc)
P.assemble()

# Create functions to split A, S

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

u_map = V.dofmap.index_map
u1_map = V1.dofmap.index_map

offset_u = u_map.local_range[0] * V.dofmap.index_map_bs + u1_map.local_range[0]
offset_u1 = offset_u + u_map.size_local * V.dofmap.index_map_bs

is_u = PETSc.IS().createStride(
    u_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=domain.comm)

is_u1 = PETSc.IS().createStride(u1_map.size_local, offset_u1, 1, comm=domain.comm)

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat, P)
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-8, atol=1e-8, max_it=100)
ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
# ksp.setNormType(2)

ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(("u", is_u), ("u1", is_u1))

ksp_u, ksp_u1 = ksp.getPC().getFieldSplitSubKSP()

ksp_u.setType("preonly")
ksp_u.getPC().setType("hypre")
ksp_u.getPC().setHYPREType("ams")

W = fem.functionspace(domain, ("Lagrange", degree))
G = discrete_gradient(W._cpp_object, V._cpp_object)
G.assemble()
ksp_u.getPC().setHYPREDiscreteGradient(G)

if degree == 1:
    cvec_0 = Function(V)
    cvec_0.interpolate(
        lambda x: np.vstack(
            (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_1 = Function(V)
    cvec_1.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_2 = Function(V)
    cvec_2.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]))
        )
    )
    ksp_u.getPC().setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )

else:
    shape = (domain.geometry.dim,)
    Q = fem.functionspace(domain, ("Lagrange", degree, shape))
    Pi = interpolation_matrix(Q._cpp_object, V._cpp_object)
    Pi.assemble()
    ksp_u.getPC().setHYPRESetInterpolations(dim=domain.geometry.dim, ND_Pi_Full=Pi)

ksp_u.getPC().setHYPREAMSSetInteriorNodes(interior_nodes_array.x.petsc_vec)

opts = PETSc.Options()
opts[f"{ksp_u.prefix}pc_hypre_ams_cycle_type"] = 13
opts[f"{ksp_u.prefix}pc_hypre_ams_tol"] = 0
opts[f"{ksp_u.prefix}pc_hypre_ams_max_iter"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_theta"] = 0.25
opts[f"{ksp_u.prefix}pc_hypre_ams_print_level"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_alpha_options"] = "10,1,6,6,4"
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_options"] = "10,1,6,6,4"
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_type"] = 2
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_weight"] = 1.0
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_times"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_omega"] = 1.0
opts[f"{ksp_u.prefix}pc_hypre_ams_projection_frequency"] = 20


ksp_u.setFromOptions()

ksp_u1.setType("preonly")
ksp_u1.getPC().setType("gamg")

ksp_u1.setFromOptions()

ksp.setUp()
ksp_u.getPC().setUp()
ksp_u1.getPC().setUp()


# ksp = PETSc.KSP().create(domain.comm)
# ksp.setOperators(A_mat)
# ksp.setType("preonly")

# pc = ksp.getPC()
# pc.setType("lu")
# pc.setFactorSolverType("mumps")


ksp.setMonitor(my_monitor)

u_n_prev = u_n.copy()

sol = A_mat.createVecRight()

ksp.solve(b, sol)


uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

uh.x.array[:offset] = sol.array_r[:offset]
uh1.x.array[:(len(sol.array_r) - offset)] = sol.array_r[offset:]

uh.x.scatter_forward()
uh1.x.scatter_forward()

u_n.x.array[:] = uh.x.array
u_n1.x.array[:] = uh1.x.array

u_n.x.scatter_forward()
u_n1.x.scatter_forward()

reason = ksp.getConvergedReason()
par_print(comm, f"Converged reason: {reason}")

res = ksp.getResidualNorm()
par_print(comm, f"Final residual: {res}")


vector_vis = fem.functionspace(
    domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
)

A_vis = Function(vector_vis)
A_vis.interpolate(u_n)
A_file = VTXWriter(domain.comm, "A_field_interior.bp", A_vis, "BP4")
A_file.write(t.expression().value)

B = curl(u_n)
B_vis = Function(vector_vis)
Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
B_vis.interpolate(Bexpr)
B_file = VTXWriter(domain.comm, "B_field_interior.bp", B_vis, "BP4")
B_file.write(t.expression().value)

da_dt = (u_n - u_n_prev) / dt
E = - grad(u_n1) - da_dt
E_vis = Function(vector_vis)
Eexpr = fem.Expression(E, vector_vis.element.interpolation_points)
E_vis.interpolate(Eexpr)
E_file = VTXWriter(domain.comm, "E_field_interior.bp", E_vis, "BP4")
E_file.write(t.expression().value)

J_ind = sigma * E
J_vis = Function(vector_vis)
Jexpr = fem.Expression(J_ind, vector_vis.element.interpolation_points)
J_vis.interpolate(Jexpr)
J_file = VTXWriter(domain.comm, "J_field_interior.bp", J_vis, "BP4")
J_file.write(t.expression().value)

u_n1_file = VTXWriter(domain.comm, "u_n1_field_interior.bp", u_n1, "BP4")
u_n1_file.write(t.expression().value)

output_freq = 10

for n in range(300):

    par_print(comm, "\n")

    t.expression().value += d_t
    par_print(comm, f"Time step {n+1}: t = {t.expression().value}")
    ksp_u.getPC().HYPREAMSResetSolveCounter()

    u_n_prev = u_n.copy()

    high_expr = fem.Expression(
    V_in * ufl.sin(omega * t), V1.element.interpolation_points
    )
    high_func.interpolate(high_expr)

    bc_ex2 = fem.dirichletbc(high_func, bdofs2)

    bc = [bc_ex, bc_ex1, bc_ex2, bc_ex3]
    
    uh.x.array[:] = 0
    uh1.x.array[:] = 0

    b = assemble_vector(L, kind=PETSc.Vec.Type.MPI)
    bcs1 = bcs_by_block(extract_function_spaces(L), bc)
    apply_lifting(b, a, bcs=bcs1)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = bcs_by_block(extract_function_spaces(L), bc)
    set_bc(b, bcs0)


    sol = A_mat.createVecRight()
    par_print(comm, f"Norm of b: {b.norm()}")

    ksp.solve(b, sol)

    par_print(comm, f"Norm of solution: {sol.norm()}")

    uh.x.array[:offset] = sol.array_r[:offset]
    uh1.x.array[:(len(sol.array_r) - offset)] = sol.array_r[offset:]

    uh.x.scatter_forward()
    uh1.x.scatter_forward()

    u_n.x.array[:] = uh.x.array
    u_n1.x.array[:] = uh1.x.array

    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()

    reason = ksp.getConvergedReason()
    par_print(comm, f"Converged reason: {reason}")

    res = ksp.getResidualNorm()
    par_print(comm, f"Final residual: {res}")

    par_print(comm, f"L2 norm of u_n is {L2_norm(u_n)}")
    par_print(comm, f"L2 norm of u_n1 is {L2_norm(u_n1)}")

    B = curl(u_n)
    da_dt = (u_n - u_n_prev) / dt
    E = -grad(u_n1) - da_dt
    J_ind = sigma * E

    par_print(comm, f"L2 norm of B is {L2_norm(B)}")
    par_print(comm, f"L2 norm of E is {L2_norm(E)}")
    par_print(comm, f"L2 norm of J is {L2_norm(J_ind)}")


    if (n + 1) % output_freq == 0:   
        A_vis.interpolate(u_n)
        A_file.write(t.expression().value)

        B = curl(u_n)
        Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
        B_vis.interpolate(Bexpr)
        B_file.write(t.expression().value)

        da_dt = (u_n - u_n_prev) / dt
        E = -grad(u_n1) - da_dt
        Eexpr = fem.Expression(E, vector_vis.element.interpolation_points)
        E_vis.interpolate(Eexpr)
        E_file.write(t.expression().value)

        J_ind = sigma * E
        Jexpr = fem.Expression(J_ind, vector_vis.element.interpolation_points)
        J_vis.interpolate(Jexpr)
        J_file.write(t.expression().value)

        u_n1_file.write(t.expression().value)

A_file.close()
B_file.close()
E_file.close()
J_file.close()
u_n1_file.close()
