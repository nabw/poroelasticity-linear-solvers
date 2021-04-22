from petsc4py import PETSc
from dolfin import *
from lib.MeshCreation import generate_cube, generate_boundary_measure
from lib.Parser import Parser
from lib.Printing import parprint
from time import time
parameters["mesh_partitioner"] = "ParMETIS"

initial_time = time()
parser = Parser()
degree_s = 2
Nelements = 10
refinements = 0
if parser.options.N:
    Nelements = parser.options.N
if parser.options.refinements:
    refinements = parser.options.refinements
side_length = 1e-2
mesh, markers, XP, XM, YP, YM, ZP, ZM = generate_cube(
    Nelements, side_length, refinements)


neumann_solid_markers = [XP, YP, ZP]
neumann_fluid_markers = [XM, YM]

dsNs = generate_boundary_measure(mesh, markers, neumann_solid_markers)
dsNf = generate_boundary_measure(mesh, markers, neumann_fluid_markers)

# Set up load terms
fs_vol = ff_vol = lambda t: Constant((0., 0., 0.))


def p_source(t): return Constant(0.0)


def ff_sur(t):
    return Constant(-1e3 * 0.1 * (1 - exp(-(t**2) / 0.25))) * FacetNormal(mesh)


def fs_sur(t):
    return Constant(-1e3 * 0.9 * (1 - exp(-(t**2) / 0.25))) * FacetNormal(mesh)


mu_f = 0.035
rhof = 1e3
rhos = 1e3
phi0 = 0.1
mu_s = 4000
lmbda = 700
ks = 1e6
kf = 1e-7
dt = 0.1
t0 = 0.0
tf = 0.1
maxiter = 1000
betas = -0.5
betaf = 0.
betap = 1.

# FE space
V = FunctionSpace(mesh,  VectorElement('CG', tetrahedron, degree_s))
parprint("Dofs = {}".format(V.dim()))
sol = Function(V)
us_nm1 = Function(V)
us_nm2 = Function(V)

# BCs
bcs = [DirichletBC(V.sub(0), Constant(0), markers, XM),
       DirichletBC(V.sub(1), Constant(0), markers, YM),
       DirichletBC(V.sub(2), Constant(0), markers, ZM)]


def fs_sur(t): return Constant(-1e3 * 0.9 * (1 - exp(-(t**2) / 0.25))) * FacetNormal(mesh)


def ff_sur(t): return Constant(-1e3 * 0.1 * (1 - exp(-(t**2) / 0.25))) * FacetNormal(mesh)


# params
rhof = rhos = 1e3
mu_f = 0.035
phi0 = 0.1
phis = 1 - phi0
mu_s = 4000
lmbda = 700
ks = 1e6
kf = 1e-7
ikf = inv(kf)

# time
t0 = 0
t = t0
tf = 0.1
dt = 1e-1
idt = 1/dt


# Compute A, P and b
us = TrialFunction(V)
v = TestFunction(V)


def hooke(ten):
    return 2 * mu_s * ten + lmbda * tr(ten) * Identity(3)


def eps(vec):
    return sym(grad(vec))


# First base matrix
a_s = (Constant(rhos / dt**2) * phis * dot(us, v)
       + inner(hooke(eps(us)), eps(v))
       - phi0 ** 2 * dot(inv(kf) * (- Constant(1. / dt) * us), v)) * dx

# Rhs
t = 0.1
rhs_s_n = dot(fs_sur(t), v) * dsNs + phis * \
    rhos * dot(fs_vol(t), v) * dx
lhs_s_n = dot(rhos * idt**2 * phis * (-2. * us_nm1 + us_nm2), v) * \
    dx - phi0**2 * dot(ikf * (- idt * (- us_nm1)), v) * dx
r_s = rhs_s_n - lhs_s_n

A = PETScMatrix()
b = PETScVector()
tt = time()
assemble(a_s, tensor=A)
assemble(r_s, tensor=b)
for bc in bcs:
    bc.apply(A, b)


def build_nullspace(V, x):
    # Function to build null space for 2D elasticity
    #
    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(6)]

    Z_rot = [Expression(('0', 'x[2]', '-x[1]'), degree=1),
             Expression(('-x[2]', '0', 'x[0]'), degree=1),
             Expression(('x[1]', '-x[0]', '0'), degree=1)]

    # Build translational null space basis - 2D problem
    #
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0)
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0)

    u = Function(V)
    for i in range(3):
        u.interpolate(Z_rot[i])
        nullspace_basis[3+i].set_local(u.vector())

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis


null_space = build_nullspace(V, sol.vector())
null_space.orthogonalize(b)

# Attach near nullspace to matrix
A.set_near_nullspace(null_space)
Amat = A.mat()
parprint("Assembled in {}s".format(time() - tt))

tt = time()
ksp = PETSc.KSP().create()
ksp.setOperators(Amat, Amat)
pc = ksp.getPC()
ksp.setOptionsPrefix("s_")
pc.setFromOptions()
ksp.setFromOptions()
ksp.solve(b.vec(), sol.vector().vec())
parprint("Solved in {} iterations in {}s".format(ksp.getIterationNumber(), time() - tt))

# Weak scaling (100k per core):
# -np 1 -N 16: Solves in 1.12s
# -np 2 -N 20: Solves in 1.49s
# -np 4 -N 25: Solves in 2.61s
# -np 6 -N 29: Solves in 3.66s
