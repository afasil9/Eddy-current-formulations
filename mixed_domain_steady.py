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
    curl,
    Measure,
)
from dolfinx.fem import Function, form
import numpy as np
from basix.ufl import element
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from utils import par_print
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import GhostMode
from utils import my_monitor, interpolate_by_tags, L2_norm
from dolfinx import default_scalar_type
from dolfinx.mesh import create_submesh
from utils import convert_facet_tags
from dolfinx.mesh import locate_entities_boundary


degree = 1

with XDMFFile(MPI.COMM_WORLD, "copper_rod.xdmf", "r") as xdmf:
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
    "bottom_surface": 4,
}

vol_ids = {"copper": 1, "air": 2}


const = fem.functionspace(domain, ("DG", 0))  # Piecewise constant function space

sigma = fem.Function(const)
nu = fem.Function(const)

mu = 4e-7 * np.pi

sigma_air = fem.Constant(domain, default_scalar_type(0.0))
sigma_copper = fem.Constant(domain, default_scalar_type(5.96e7))
nu_value = fem.Constant(domain, default_scalar_type(1 / mu))

sigma_values = {
    vol_ids["air"]: sigma_air,
    vol_ids["copper"]: sigma_copper,
}

nu_values = {vol_ids["air"]: nu_value, vol_ids["copper"]: nu_value}

interpolate_by_tags(sigma, sigma_values, ct)
interpolate_by_tags(nu, nu_values, ct)


gdim = domain.geometry.dim
facet_dim = gdim - 1

copper_cells = ct.find(vol_ids["copper"])

submesh_copper, subdomain_copper_to_domain = create_submesh(domain, tdim, copper_cells)[
    :2
]
entity_maps = [subdomain_copper_to_domain]


nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = functionspace(domain, nedelec_elem)

lagrange_elem = element("Lagrange", submesh_copper.basix_cell(), degree)
V1 = functionspace(submesh_copper, lagrange_elem)


copper_ft = convert_facet_tags(submesh_copper, subdomain_copper_to_domain, ft)
submesh_copper.topology.create_connectivity(fdim, tdim)

upper_facets = copper_ft.find(boundary_tags["upper_surface"])
bdofs2 = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=upper_facets)
high_func = fem.Function(V1)
high_func.x.array[:] = 10.0
bc_upper = fem.dirichletbc(high_func, bdofs2)

lower_facets = copper_ft.find(boundary_tags["bottom_surface"])
bdofs3 = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=lower_facets)
low_func = fem.Function(V1)
low_func.x.array[:] = 0.0
bc_lower = fem.dirichletbc(low_func, bdofs3)


boundary_tags_total = (1, 2, 4)
boundary_facets_V = np.concatenate([ft.find(tag) for tag in boundary_tags_total])

bdofs0 = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=boundary_facets_V)
u_bc_V = fem.Function(V)
u_bc_V.x.array[:] = 0
bc_outer = fem.dirichletbc(u_bc_V, bdofs0)

bc = [bc_outer, bc_upper, bc_lower]

u_n = fem.Function(V)
u_n1 = fem.Function(V1)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

dx = Measure("dx", domain=domain, subdomain_data=ct)

whole = (1, 2)

a00 = ufl.inner(nu * curl(u), curl(v)) * dx(whole)
a01 = ufl.inner(sigma * grad(u1), v) * dx(vol_ids["copper"])
a11 = ufl.inner(sigma * ufl.grad(u1), ufl.grad(v1)) * dx(vol_ids["copper"])

zero_vec = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0)))
L0 = ufl.inner(zero_vec, v) * dx(whole)
L1 = fem.Constant(domain, PETSc.ScalarType(0.0)) * v1 * dx(vol_ids["copper"])

a = form([[a00, a01], [None, a11]], entity_maps=entity_maps)

comm = MPI.COMM_WORLD
par_print(comm, "Assembling system matrix...")

A_mat = assemble_matrix(a, bcs=bc)
A_mat.assemble()


L = form([L0, L1], entity_maps=entity_maps)

b = assemble_vector(L)
bcs1 = bcs_by_block(extract_function_spaces(a, 1), bc)
apply_lifting(b, a, bcs=bcs1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bcs0 = bcs_by_block(extract_function_spaces(L), bc)
set_bc(b, bcs0)

a_p = form([[a00, None], [None, a11]], entity_maps=entity_maps)

P = assemble_matrix(a_p, bcs=bc)
P.assemble()

# Create functions to split A, S

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

u_map = V.dofmap.index_map
u1_map = V1.dofmap.index_map

offset_u = u_map.local_range[0] * V.dofmap.index_map_bs + u1_map.local_range[0]
offset_u1 = offset_u + u_map.size_local * V.dofmap.index_map_bs

is_u = PETSc.IS().createStride(
    u_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=domain.comm
)
is_u1 = PETSc.IS().createStride(u1_map.size_local, offset_u1, 1, comm=domain.comm)


ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat, P)
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-14, atol=1e-50, max_it=100)
ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.UPPER)
pc.setFieldSplitIS(("u", is_u), ("u1", is_u1))

ksp.setUp()

ksp_u, ksp_u1 = pc.getFieldSplitSubKSP()

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

ksp_u.getPC().setHYPRESetBetaPoissonMatrix(None)

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

ksp_u.setFromOptions()

ksp_u1.setType("preonly")
ksp_u1.getPC().setType("hypre")
ksp_u1.getPC().setHYPREType("boomeramg")

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

# opts = PETSc.Options()
# opts["mat_mumps_icntl_14"] = 80
# opts["mat_mumps_icntl_24"] = (
#     1
# )
# opts["mat_mumps_icntl_25"] = (
#     0
# )
# opts["ksp_error_if_not_converged"] = 1
# ksp.setFromOptions()

ksp.setMonitor(my_monitor)

sol = A_mat.createVecRight()

par_print(comm, "about to solve")
ksp.solve(b, sol)


reason = ksp.getConvergedReason()
par_print(comm, f"Converged reason: {reason}")

res = ksp.getResidualNorm()
par_print(comm, f"Final residual: {res}")


uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

uh.x.array[:offset] = sol.array_r[:offset]
uh1.x.array[: (len(sol.array_r) - offset)] = sol.array_r[offset:]

uh.x.scatter_forward()
uh1.x.scatter_forward()

u_n.x.array[:] = uh.x.array
u_n1.x.array[:] = uh1.x.array

u_n.x.scatter_forward()
u_n1.x.scatter_forward()

par_print(comm, f"L2 norm of solution fields {L2_norm(u_n)}")
par_print(comm, f"L2 norm of solution fields {L2_norm(u_n1)}")

t = 0.0

vector_vis = fem.functionspace(
    domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
)

A_vis = Function(vector_vis)
A_file = VTXWriter(domain.comm, "A_field_submesh.bp", A_vis, "BP4")
A_vis.interpolate(u_n)
A_file.write(t)

B = curl(u_n)
B_vis = Function(vector_vis)
B_file = VTXWriter(domain.comm, "B_field_submesh.bp", B_vis, "BP4")
Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
B_vis.interpolate(Bexpr)
B_file.write(t)

u_n_prev = u_n.copy()

V_submesh = functionspace(submesh_copper, nedelec_elem)
u_n_submesh = Function(V_submesh)


smsh_cell_imap = submesh_copper.topology.index_map(tdim)
smsh_cells = np.arange(smsh_cell_imap.size_local + smsh_cell_imap.num_ghosts)
parent_cells = subdomain_copper_to_domain.sub_topology_to_topology(
    smsh_cells, inverse=False
)

u_n_submesh.interpolate(u_n, cells0=parent_cells, cells1=smsh_cells)

E = -grad(u_n1)

Submesh_DG = functionspace(
    submesh_copper, ("DG", degree + 1, (submesh_copper.geometry.dim,))
)

E_vis = Function(Submesh_DG)
E_expr = fem.Expression(E, Submesh_DG.element.interpolation_points)
E_vis.interpolate(E_expr)

E_file = VTXWriter(domain.comm, "E_field_submesh.bp", E_vis, "BP4")
E_file.write(t)

DG0_submesh = functionspace(submesh_copper, ("DG", 0))
sigma_submesh = Function(DG0_submesh)
sigma_submesh.interpolate(sigma, cells0=parent_cells, cells1=smsh_cells)

J = sigma_submesh * E

J_vis = Function(Submesh_DG)
J_expr = fem.Expression(J, Submesh_DG.element.interpolation_points)
J_vis.interpolate(J_expr)
J_file = VTXWriter(domain.comm, "J_field_submesh.bp", J_vis, engine="BP4")
J_file.write(t)

u_n1_file = VTXWriter(domain.comm, "u_n1_field_submesh.bp", u_n1, "BP4")
u_n1_file.write(t)


par_print(comm, f"B norm is {L2_norm(B)}")
par_print(comm, f"E norm is {L2_norm(E)}")
par_print(comm, f"J norm is {L2_norm(J)}")
par_print(comm, f"u_n1 norm is {L2_norm(u_n1)}")

exit()

W_comp = fem.functionspace(domain, ("Lagrange", degree))
phi = ufl.TrialFunction(W_comp)
q = ufl.TestFunction(W_comp)

def all_boundary(x):
    return np.full(x.shape[1], True)
domain.topology.create_connectivity(tdim - 1, tdim)
bndry_facets = locate_entities_boundary(domain, tdim - 1, all_boundary)
bndry_dofs_W = fem.locate_dofs_topological(W_comp, tdim - 1, bndry_facets)

zero_W = fem.Function(W_comp)
zero_W.x.array[:] = 0.0
bc_W = fem.dirichletbc(zero_W, bndry_dofs_W)

# Matrix is domain-only -> no entity_maps
a_comp = form(ufl.inner(ufl.grad(phi), ufl.grad(q)) * dx)

# RHS uses u_n1 (submesh) + q (domain) -> MUST pass entity_maps
L_comp = form(
    ufl.inner(sigma * ufl.grad(u_n1), ufl.grad(q)) * dx(vol_ids["copper"]),
    entity_maps=entity_maps,
)

A_comp = assemble_matrix(a_comp, bcs=[bc_W])
A_comp.assemble()

b_comp = assemble_vector(L_comp)
apply_lifting(b_comp, [a_comp], bcs=[[bc_W]])
b_comp.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b_comp, [bc_W])

compat_vec_norm = b_comp.norm()
print(f"Compatibility vector norm ||(J,grad q)|| = {compat_vec_norm:.6e}")

phi_h = fem.Function(W_comp)
ksp_comp = PETSc.KSP().create(domain.comm)
ksp_comp.setOperators(A_comp)
ksp_comp.setType("preonly")
ksp_comp.getPC().setType("lu")
ksp_comp.getPC().setFactorSolverType("mumps")
ksp_comp.solve(b_comp, phi_h.x.petsc_vec)
phi_h.x.scatter_forward()

print(f"Compatibility KSP converged with reason {ksp_comp.getConvergedReason()}")
print(f"Compatibility KSP iteration count: {ksp_comp.getIterationNumber()}")
print(f"Compatibility KSP residual norm: {ksp_comp.getResidualNorm():.6e}")

def global_scalar(frm):
    local_val = fem.assemble_scalar(frm)
    return domain.comm.allreduce(local_val, op=MPI.SUM)

grad_part_sq = global_scalar(form(ufl.inner(ufl.grad(phi_h), ufl.grad(phi_h)) * dx))
j_norm_sq = global_scalar(
    form(
        ufl.inner(sigma * ufl.grad(u_n1), sigma * ufl.grad(u_n1)) * dx(vol_ids["copper"]),
        entity_maps=entity_maps,
    )
)

grad_part = np.sqrt(max(grad_part_sq, 0.0))
j_norm = np.sqrt(max(j_norm_sq, 0.0))
ratio = grad_part / max(j_norm, 1e-30)

print(f"||J_grad_part|| = {grad_part:.6e}")
print(f"||J||           = {j_norm:.6e}")
print(f"relative incompatible part = {ratio:.6e}")


