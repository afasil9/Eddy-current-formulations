#%%
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from dolfinx import fem, default_scalar_type
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.mesh import GhostMode
from basix.ufl import element
from dolfinx.fem import form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from utils import L2_norm, my_monitor, interpolate_by_tags, par_print

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

sigma_air = fem.Constant(domain, default_scalar_type(0.0))
sigma_copper = fem.Constant(domain, default_scalar_type(5.96e7))

sigma_values = {
    vol_ids["air"]: sigma_air,
    vol_ids["copper"]: sigma_copper,
}

interpolate_by_tags(sigma, sigma_values, ct)


degree = 1

# Scalar CG space for u_n1
lagrange_elem = element("Lagrange", domain.basix_cell(), degree)
V1 = fem.functionspace(domain, lagrange_elem)

# Dirichlet boundary conditions on V1
gdim = domain.geometry.dim
facet_dim = gdim - 1


outer_boundary_facets = ft.find(boundary_tags["cube_boundary"])

bdofs1 = fem.locate_dofs_topological(
    V1, entity_dim=facet_dim, entities=outer_boundary_facets
)
u_bc_V1 = fem.Function(V1)
u_bc_V1.x.array[:] = 0.0
bc_ex1 = fem.dirichletbc(u_bc_V1, bdofs1)

# Upper (facet tag 3) -> 10.0
upper_facets = ft.find(boundary_tags["upper_surface"])
bdofs2 = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=upper_facets)
high_func = fem.Function(V1)
high_func.x.array[:] = 10.0
bc_ex2 = fem.dirichletbc(high_func, bdofs2)

# Lower (facet tag 1) -> 0.0
lower_facets = ft.find(boundary_tags["bottom_surface"])
bdofs3 = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=lower_facets)
low_func = fem.Function(V1)
low_func.x.array[:] = 0.0
bc_ex3 = fem.dirichletbc(low_func, bdofs3)

bcs = [bc_ex1, bc_ex2, bc_ex3]

# Variational problem: ∫ σ ∇u · ∇v dx = 0
u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)
dx = ufl.Measure("dx", domain, subdomain_data=ct)


lhs = ufl.inner(sigma * ufl.grad(u1), ufl.grad(v1)) * dx
rhs = fem.Constant(domain, PETSc.ScalarType(0.0)) * v1 * dx

# bcs = [bc_ex1]  # Without upper surface BC for current source
# current_magnitude = 10.0
# ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
# n = ufl.FacetNormal(domain)
# rhs = fem.Constant(domain, PETSc.ScalarType(0.0)) * v1 * dx + current_magnitude * v1 * ds(boundary_tags["upper_surface"])

a = form(lhs)
L = form(rhs)

# Assemble system
A = assemble_matrix(a, bcs=bcs)
A.assemble()
b = assemble_vector(L)
apply_lifting(b, [a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs)


# Solve with PETSc
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)

ksp.setType("gmres")
ksp.setTolerances(rtol=1e-15, atol=1e-50, max_it=1000)

pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("boomeramg")


# ksp = PETSc.KSP().create(domain.comm)
# ksp.setOperators(A)
# ksp.setType("preonly")

# pc = ksp.getPC()
# pc.setType("lu")
# pc.setFactorSolverType("mumps")

# opts = PETSc.Options()  # type: ignore
# opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
# opts["mat_mumps_icntl_24"] = (
#     1  # Option to support solving a singular matrix (pressure nullspace)
# )
# opts["mat_mumps_icntl_25"] = (
#     0  # Option to support solving a singular matrix (pressure nullspace)
# )
# opts["ksp_error_if_not_converged"] = 1
# ksp.setFromOptions()

ksp.setMonitor(my_monitor)

pc.setUp()
ksp.setUp()

u_n1 = fem.Function(V1)
ksp.solve(b, u_n1.x.petsc_vec)
u_n1.x.scatter_forward()

reason = ksp.getConvergedReason()
print(f"KSP converged with reason {reason}")


# Output
t = 0.0
u_n1_file = VTXWriter(domain.comm, "V_field.bp", u_n1, "BP4")
u_n1_file.write(t)
u_n1_file.close()

comm = MPI.COMM_WORLD
par_print(comm, f"L2 norm of u_n1 field {L2_norm(u_n1)}")

res = b - A * u_n1.x.petsc_vec

E = -ufl.grad(u_n1)
J = sigma * E

vector_vis = fem.functionspace(
    domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
)

E_vis = fem.Function(vector_vis)
Eexpr = fem.Expression(E, vector_vis.element.interpolation_points)
E_vis.interpolate(Eexpr)
E_file = VTXWriter(domain.comm, "E_field.bp", E_vis, "BP4")
E_file.write(t)
E_file.close()

J_vis = fem.Function(vector_vis)
Jexpr = fem.Expression(J, vector_vis.element.interpolation_points)
J_vis.interpolate(Jexpr)
J_file = VTXWriter(domain.comm, "J_field.bp", J_vis, "BP4")
J_file.write(t)
J_file.close()

n = ufl.FacetNormal(domain)
J = -sigma * ufl.grad(u_n1)
ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

I_form = ufl.dot(J, n) * ds(boundary_tags["bottom_surface"])
surface_current = fem.assemble_scalar(fem.form(I_form))

par_print(comm, f"Current is: {surface_current:.6e} A")

# %%
