#%%
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from dolfinx import fem, default_scalar_type
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.mesh import GhostMode
from basix.ufl import element
from dolfinx.fem import form, functionspace, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from utils import my_monitor, interpolate_by_tags
from dolfinx.cpp.fem.petsc import (
    discrete_gradient,
    interpolation_matrix,
)
from utils import par_print, L2_norm

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
comm = MPI.COMM_WORLD
degree = 1

const = fem.functionspace(domain, ("DG", 0))  # Piecewise constant function space

nu = fem.Function(const)
sigma = fem.Function(const)

mu = 4e-7 * np.pi

sigma_air = fem.Constant(domain, default_scalar_type(0.0))
sigma_copper = fem.Constant(domain, default_scalar_type(5.96e7))
nu_value = fem.Constant(domain, default_scalar_type(1 / mu))

sigma_values = {
    vol_ids["air"]: sigma_air,
    vol_ids["copper"]: sigma_copper,
}

nu_values = {
    vol_ids["air"]: nu_value,
    vol_ids["copper"]: nu_value,
}

interpolate_by_tags(sigma, sigma_values, ct)
interpolate_by_tags(nu, nu_values, ct)

gdim = domain.geometry.dim
facet_dim = gdim - 1

dx = ufl.Measure("dx", domain, subdomain_data=ct)

# Scalar CG space for u_n1
lagrange_elem = element("Lagrange", domain.basix_cell(), degree)
V1 = fem.functionspace(domain, lagrange_elem)

outer_boundary_facets = ft.find(boundary_tags["cube_boundary"])

bdofs1 = fem.locate_dofs_topological(
    V1, entity_dim=facet_dim, entities=outer_boundary_facets
)
u_bc_V1 = fem.Function(V1)
u_bc_V1.x.array[:] = 0.0
bc_ex1 = fem.dirichletbc(u_bc_V1, bdofs1)

# Upper (facet tag 2) -> 10.0
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


lhs = ufl.inner(sigma * ufl.grad(u1), ufl.grad(v1)) * dx
rhs = fem.Constant(domain, PETSc.ScalarType(0.0)) * v1 * dx

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

ksp.setMonitor(my_monitor)

pc.setUp()
ksp.setUp()

u_n1 = fem.Function(V1)
ksp.solve(b, u_n1.x.petsc_vec)
u_n1.x.scatter_forward()

reason = ksp.getConvergedReason()
par_print(comm, f"KSP converged with reason {reason}")


# Output
t = 0.0
u_n1_file = VTXWriter(domain.comm, "u_n1_field_uncoupled.bp", u_n1, "BP4")
u_n1_file.write(t)
u_n1_file.close()

res = b - A * u_n1.x.petsc_vec

E = -ufl.grad(u_n1)
J = sigma * E

vector_vis = fem.functionspace(
    domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
)

E_vis = fem.Function(vector_vis)
Eexpr = fem.Expression(E, vector_vis.element.interpolation_points)
E_vis.interpolate(Eexpr)
E_file = VTXWriter(domain.comm, "E_field_uncoupled.bp", E_vis, "BP4")
E_file.write(t)
E_file.close()

J_vis = fem.Function(vector_vis)
Jexpr = fem.Expression(J, vector_vis.element.interpolation_points)
J_vis.interpolate(Jexpr)
J_file = VTXWriter(domain.comm, "J_field_uncoupled.bp", J_vis, "BP4")
J_file.write(t)
J_file.close()


# Solving for A field

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = functionspace(domain, nedelec_elem)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

lhs_curl = ufl.inner(nu * ufl.curl(u), ufl.curl(v)) * dx
rhs_curl = ufl.inner(J, v) * dx

a_curl = form(lhs_curl)
L_curl = form(rhs_curl)


boundary_tags_total = (1, 2, 4)
boundary_facets_V = np.concatenate([ft.find(tag) for tag in boundary_tags_total])

bdofs0 = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=boundary_facets_V)
u_bc_V = fem.Function(V)
u_bc_V.x.array[:] = 0
bc_ex = fem.dirichletbc(u_bc_V, bdofs0)

A_mat = assemble_matrix(a_curl, bcs=[bc_ex])
A_mat.assemble()
b = assemble_vector(L_curl)
apply_lifting(b, [a_curl], bcs=[[bc_ex]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs=[bc_ex])


ksp_curl = PETSc.KSP().create(domain.comm)
ksp_curl.setOperators(A_mat)
ksp_curl.setMonitor(my_monitor)
ksp_curl.setTolerances(rtol=1e-12, atol=1e-12, max_it=1000)
ksp_curl.setType("cg")
pc_curl = ksp_curl.getPC()
pc_curl.setType("hypre")
pc_curl.setHYPREType("ams")

# Build discrete gradient
V_CG = fem.functionspace(domain, ("CG", degree))._cpp_object
G = discrete_gradient(V_CG, V._cpp_object)
G.assemble()
pc_curl.setHYPREDiscreteGradient(G)

pc_curl.setHYPRESetBetaPoissonMatrix(None)

if degree == 1:
        cvec_0 = Function(V)
        cvec_0.interpolate(lambda x: np.vstack((np.ones_like(x[0]),
                                                np.zeros_like(x[0]),
                                                np.zeros_like(x[0]))))
        cvec_1 = Function(V)
        cvec_1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                                np.ones_like(x[0]),
                                                np.zeros_like(x[0]))))
        cvec_2 = Function(V)
        cvec_2.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                                np.zeros_like(x[0]),
                                                np.ones_like(x[0]))))
        pc_curl.setHYPRESetEdgeConstantVectors(cvec_0.x.petsc_vec,
                                                cvec_1.x.petsc_vec,
                                                cvec_2.x.petsc_vec)
else:   
        Vec_CG = fem.functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
        Pi = interpolation_matrix(Vec_CG._cpp_object, V._cpp_object)
        Pi.assemble()

        # Attach discrete gradient to preconditioner
        pc_curl.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)

ksp_curl.setUp()
pc_curl.setUp()


u_n = fem.Function(V)

ksp_curl.solve(b, u_n.x.petsc_vec)
u_n.x.scatter_forward()

t = 0.0
vector_vis = fem.functionspace(
    domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
)

A_vis = Function(vector_vis)
A_file = VTXWriter(domain.comm, "A_field_uncoupled.bp", A_vis, "BP4")
A_vis.interpolate(u_n)
A_file.write(t)

B = ufl.curl(u_n)
B_vis = Function(vector_vis)
B_file = VTXWriter(domain.comm, "B_field_uncoupled.bp", B_vis, "BP4")
Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
B_vis.interpolate(Bexpr)
B_file.write(t)

par_print(comm, f"B norm is {L2_norm(B)}")
par_print(comm, f"E norm is {L2_norm(E)}")
par_print(comm, f"J norm is {L2_norm(J)}")
par_print(comm, f"u_n1 norm is {L2_norm(u_n1)}")
par_print(comm, f"u_n norm is {L2_norm(u_n)}")

n = ufl.FacetNormal(domain)
J = -sigma * ufl.grad(u_n1)
ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

I_form = ufl.dot(J, n) * ds(boundary_tags["bottom_surface"])
I_surf = fem.assemble_scalar(fem.form(I_form))

par_print(comm, f"Current is: {I_surf:.6e} A")

# %%
