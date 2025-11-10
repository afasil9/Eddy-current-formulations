#%%
import numpy as np
from basix.ufl import element
from dolfinx import fem
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import (
    Function,
    dirichletbc,
    form,
    locate_dofs_topological,
    petsc,
)
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import GhostMode
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    curl,
    inner,
    variable,
)
from dolfinx import default_scalar_type

from utils import L2_norm, par_print, interpolate_by_tags


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

sigma_air = fem.Constant(domain, default_scalar_type(1e-7))
sigma_copper = fem.Constant(domain, default_scalar_type(5.96e4))
nu_value = fem.Constant(domain, default_scalar_type(1e6))

sigma_values = {
    vol_ids["air"]: sigma_air,
    vol_ids["copper"]: sigma_copper,
}

nu_values = {vol_ids["air"]: nu_value, vol_ids["copper"]: nu_value}

interpolate_by_tags(sigma, sigma_values, ct)
interpolate_by_tags(nu, nu_values, ct)




comm = MPI.COMM_WORLD
degree = 1

V_CG = fem.functionspace(domain, ("CG", degree))

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 100  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
A_space = fem.functionspace(domain, nedelec_elem)


x = SpatialCoordinate(domain)
a_n = fem.Function(A_space)

a_n_prev = a_n.copy()

A = TrialFunction(A_space)
v = TestFunction(A_space)

dx = Measure("dx", domain, subdomain_data=ct)


interior_nodes_array = fem.Function(V_CG)

interior_nodes_array.x.array[:] = 1.0
interior_nodes_array.x.scatter_forward()

dofmap = V_CG.dofmap
num_dofs_per_cell = dofmap.dof_layout.num_dofs
cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

tagged_cells = ct.find(vol_ids["copper"])

tagged_cell_dofs = cell_dofs[tagged_cells].flatten()
unique_dofs = np.unique(tagged_cell_dofs)

interior_nodes_array.x.array[unique_dofs] = 0.0
interior_nodes_array.x.scatter_forward()

# Q = fem.functionspace(domain, ("DG", 0))
# J = fem.Function(Q)
# J.x.array[:] = 0.0

# cells_inner = ct.find(vol_ids["copper"])
# J.x.array[cells_inner] = 1.0

# f = as_vector((0.0, 0.0, 1.0))

from ufl import sin, pi

freq = fem.Constant(domain, default_scalar_type(50.0))  # frequency in Hz
f = as_vector((0.0, 0.0, sin(2.0 * pi * freq * t)))

par_print(comm, f"f norm is {L2_norm(f)}")

lhs = dt * inner(nu * curl(A), curl(v)) * dx + inner(sigma * A, v) * dx
rhs = dt * inner(f, v) * dx(vol_ids["copper"]) + inner(sigma * a_n, v) * dx
# rhs = dt * J * v[2] * dx + inner(sigma * a_n, v) * dx

a = form(lhs)
L = form(rhs)

# Boundary conditions

boundary_tags_V = (boundary_tags["upper_surface"],boundary_tags["cube_boundary"],boundary_tags["bottom_surface"])
boundary_facets_V = np.concatenate([ft.find(tag) for tag in boundary_tags_V])

dofs = locate_dofs_topological(V=A_space, entity_dim=fdim, entities=boundary_facets_V)
u_bc = Function(A_space)
u_bc.x.array[:] = 0
bc = dirichletbc(u_bc, dofs)


# Solver steps
A_mat = assemble_matrix(a, bcs=[bc])
A_mat.assemble()


b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])

uh = fem.Function(A_space)

num_cells = domain.topology.index_map(domain.topology.dim).size_local

ams_opts = {
    "ksp_atol": 1e-10,
    "ksp_rtol": 1e-10,
    "ksp_type": "cg",
    "ksp_max_it": 50,
    "ksp_monitor_true_residual": None,
    "ksp_norm_type": "unpreconditioned",
    "pc_hypre_ams_cycle_type": 13,
    "pc_hypre_ams_tol": 0.0,  # Default is 1e-6 but we set it to 0.0 for AMS to be used as preconditioner
    "pc_hypre_ams_max_iter": 1,  # Set to 1 to use AMS as a preconditioner
    "pc_hypre_ams_print_level": 1,
    "pc_hypre_ams_amg_alpha_options": "10,1,6,6,4",
    "pc_hypre_ams_amg_beta_options": "10,1,6,6,4",
    "pc_hypre_ams_projection_frequency": 25,
    "pc_hypre_ams_relax_type": 2,
    "pc_hypre_ams_relax_weight": 1.0,
    "pc_hypre_ams_relax_times": 1,
    "pc_hypre_ams_omega": 1.0,
}

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

G = discrete_gradient(V_CG._cpp_object, A_space._cpp_object)
G.assemble()
pc.setHYPREDiscreteGradient(G)

# pc.setHYPREAMSSetInteriorNodes(interior_nodes_array.x.petsc_vec)

if degree == 1:
    cvec_0 = Function(A_space)
    cvec_0.interpolate(
        lambda x: np.vstack(
            (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_1 = Function(A_space)
    cvec_1.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_2 = Function(A_space)
    cvec_2.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]))
        )
    )

    pc.setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )
else:
    Vec_CG = fem.functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
    Pi = interpolation_matrix(Vec_CG._cpp_object, A_space._cpp_object)
    Pi.assemble()

    # Attach discrete gradient to preconditioner
    pc.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)


ksp.setFromOptions()
ksp.setUp()
pc.setUp()

ksp.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()

a_n.x.array[:] = uh.x.array
a_n.x.scatter_forward()

par_print(comm, f"Norm of a_n: {L2_norm(a_n)}")
par_print(comm, f"Norm of B: {L2_norm(curl(a_n))}")

vector_vis = fem.functionspace(
    domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
)

A_vis = Function(vector_vis)
A_file = VTXWriter(domain.comm, "A_field_aform.bp", A_vis, "BP4")
A_vis.interpolate(a_n)
A_file.write(t)

B = curl(a_n)
B_vis = Function(vector_vis)
B_file = VTXWriter(domain.comm, "B_field_aform.bp", B_vis, "BP4")
Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)

B_vis.interpolate(Bexpr)
B_file.write(t)

da_dt = (a_n - a_n_prev) / dt
E = -da_dt
E_vis = Function(vector_vis)
Eexpr = fem.Expression(E, vector_vis.element.interpolation_points)
E_vis.interpolate(Eexpr)
E_file = VTXWriter(domain.comm, "E_field_aform.bp", E_vis, "BP4")
E_file.write(t)

J_ind = sigma * E
J_vis = Function(vector_vis)
Jexpr = fem.Expression(J_ind, vector_vis.element.interpolation_points)
J_vis.interpolate(Jexpr)
J_file = VTXWriter(domain.comm, "J_field_aform.bp", J_vis, "BP4")
J_file.write(t)


output_freq = 5  # Frequency of writing output

for n in range(num_steps):

    par_print(comm, "\n")
    a_n_prev = a_n.copy()

    t.expression().value += d_t
    par_print(comm, f"n is {n}")
    par_print(comm, f"t is {t.expression().value}")

    par_print(comm, f"f norm is {L2_norm(f)}")

    with b.localForm() as loc_b:
        loc_b.set(0)
    petsc.assemble_vector(b, L)

    petsc.apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh.x.array[:] = 0.0

    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    res = b - A_mat * uh.x.petsc_vec
    par_print(comm, f"Residual norm: {res.norm()}")

    a_n.x.array[:] = uh.x.array
    a_n.x.scatter_forward()

    iterations = ksp.getIterationNumber()
    par_print(comm, f"Iteration count is {iterations}")
    
    reason = ksp.getConvergedReason()
    par_print(comm, f"KSP converged reason: {reason}")


    B = curl(a_n)
    da_dt = (a_n - a_n_prev) / dt
    E = -da_dt
    J_ind = sigma * E

    par_print(comm, f"Norm of a_n: {L2_norm(a_n)}")
    par_print(comm, f"Norm of B: {L2_norm(B)}")
    par_print(comm, f"Norm of E: {L2_norm(E)}")
    par_print(comm, f"Norm of J: {L2_norm(J_ind)}")

    if (n + 1) % output_freq == 0:
        par_print(comm, "Writing output ")   
        B = curl(a_n)

        B_vis.interpolate(Bexpr)
        B_file.write(t)

        da_dt = (a_n - a_n_prev) / dt
        E = -da_dt
        E_vis.interpolate(Eexpr)
        E_file.write(t)

        J_ind = sigma * E
        J_vis.interpolate(Jexpr)
        J_file.write(t)

