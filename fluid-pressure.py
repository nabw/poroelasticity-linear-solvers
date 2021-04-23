from petsc4py import PETSc
from dolfin import *
from lib.MeshCreation import generate_cube, generate_boundary_measure
from lib.Parser import Parser
from lib.Printing import parprint
from time import time
parameters["mesh_partitioner"] = "ParMETIS"

initial_time = time()
parser = Parser()
degree_f = 2
degree_p = 1
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


mu_f = 0.035
rhof = 1e3
rhos = 1e3
phi0 = 0.1
phis = 1-phi0
mu_s = 4000
lmbda = 700
ks = 1e6
kf = 1e-7
dt = 0.1
t0 = 0.0
tf = 0.1
ikf = inv(kf)

# time
t0 = 0
tf = 0.1
dt = 1e-1
idt = 1/dt

# FE space
V = FunctionSpace(mesh,
                  MixedElement(VectorElement('CG', mesh.ufl_cell(), degree_f),
                               FiniteElement('CG', mesh.ufl_cell(), degree_p)))

parprint("Dofs = {}".format(V.dim()))
sol = Function(V)
uf_nm1, p_nm1 = sol.split(True)

# BCs
bcs = [DirichletBC(V.sub(0), Constant((0, 0, 0)), markers, ZM),
       DirichletBC(V.sub(0), Constant((0, 0, 0)), markers, ZP)]

# Set up load terms


def ff_vol(t): return Constant((0., 0., 0.))


def p_source(t): return Constant(0.0)


def ff_sur(t): return Constant(-1e3 * 0.1 * (1 - exp(-(t**2) / 0.25))) * FacetNormal(mesh)


# Compute A, P and b
vf, p = TrialFunctions(V)
w, q = TestFunctions(V)


def eps(vec):
    return sym(grad(vec))


# First base matrix
a_f = (rhof * idt * phi0 * dot(vf, w)
       + 2. * mu_f *
       inner(phi0 * eps(vf), eps(w))
       - p * div(phi0 * w)
       + phi0 ** 2 * dot(ikf * vf, w)) * dx

a_p = (phis**2 * idt / ks * p * q
       + div(phi0 * vf) * q) * dx

# Rhs
t = 0.1
# Compute fluid residual
rhs_f_n = dot(ff_sur(t), w) * dsNf + phi0 * \
    rhof * dot(ff_vol(t), w) * dx
lhs_f = dot(rhof * idt * phi0 * (- uf_nm1), w)
lhs_f_n = lhs_f * dx
r_f = rhs_f_n - lhs_f_n
# Compute pressure residual
rhs_p_n = 1 / rhof * p_source(t) * q * dx
M_p = phis**2 / Constant(ks * dt) * (- p_nm1) * q
lhs_p_n = (M_p) * dx
r_p = rhs_p_n - lhs_p_n

A = PETScMatrix()
b = PETScVector()
tt = time()
assemble(a_f+a_p, tensor=A)
assemble(r_f+r_f, tensor=b)
for bc in bcs:
    bc.apply(A, b)


Amat = A.mat()
parprint("Assembled in {}s".format(time() - tt))

tt = time()
ksp = PETSc.KSP().create()
ksp.setOperators(Amat, Amat)
pc = ksp.getPC()
ksp.setOptionsPrefix("fp_")

# Set fieldsplit dofs
pc.setType('fieldsplit')
is_f = PETSc.IS().createGeneral(V.sub(0).dofmap().dofs())
is_p = PETSc.IS().createGeneral(V.sub(1).dofmap().dofs())
pc.setFieldSplitIS((None, is_f))
pc.setFieldSplitIS((None, is_p))
pc.setFromOptions()
ksp.setFromOptions()
ksp.solve(b.vec(), sol.vector().vec())
parprint("Solved in {} iterations in {}s".format(ksp.getIterationNumber(), time() - tt))

# Weak scaling (100k per core):
# -np 1 -N 16: Solves in 1.12s
# -np 2 -N 20: Solves in 1.49s
# -np 4 -N 25: Solves in 2.61s
# -np 6 -N 29: Solves in 3.66s
