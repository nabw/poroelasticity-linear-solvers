from dolfin import *
from lib.MeshCreation import generate_footing_square, generate_boundary_measure
from lib.Poromechanics import Poromechanics
from lib.Parser import Parser
from time import time
import numpy as np
parameters["mesh_partitioner"] = "ParMETIS"

initial_time = time()
parser = Parser()
Nelements = 10
refinements = 0
if parser.options.N:
    Nelements = parser.options.N
if parser.options.refinements:
    refinements = parser.options.refinements
length = 64
mesh, markers, LEFT, RIGHT, TOP, BOTTOM, NONE = generate_footing_square(
    Nelements, length, refinements)


neumann_solid_markers = [TOP]  # All others get weakly 0'd.
neumann_fluid_markers = []
dsNs = generate_boundary_measure(mesh, markers, neumann_solid_markers)
dsNf = generate_boundary_measure(mesh, markers, neumann_fluid_markers)

# Set up load terms
fs_vol = ff_vol = fs_sur = lambda t: Constant((0., 0.))


def p_source(t): return Constant(0)


def ff_sur(t): return Constant((0, 0))


def fs_sur(t):
    # return Expression(("0", "abs(x[0]-L)<L/2?-t*1e5:0"), t=min(0.5, t), L=length / 2, degree=4)
    return Expression(("0", "abs(x[0]-L)<L/2?(-val):0"), L=length / 2, val=min(t, 1.0)*1e5, degree=1)


E = 3e4
nu = 0.2
mu_s = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
parameters = {"mu_f": 1e-3,
              "rhof": 1e3,
              "rhos": 500,
              "phi0": 1e-3,
              "mu_s": mu_s,
              "lmbda": lmbda,
              "ks": 1e6,
              "kf": 1e-7,
              "dt": 0.1,
              "t0": 0.0,
              "tf": 0.1,
              "fe degree solid": 2,
              "fe degree fluid": 2,
              "fe degree pressure": 1,
              "maxiter": 1000,
              "output solutions": False,
              "output name": "footing",
              "betas": -0.5,
              "betaf": 0.,
              "betap": 1.,
              "solver rtol": 1e-6,
              "solver atol": 1e-4,
              "solver maxiter": 500,
              "solver monitor": False,
              "solver type": "gmres",  # cg, gmres, aar
              "pc type": "undrained",  # diagonal, undrained, diagonal 3-way
              "inner ksp type": "gmres",  # preonly, gmres, cg, bicgstab,
              "inner pc type": "hypre",  # bjacobi, ilu, hypre, lu, gamg, asm
              "inner atol": 0,
              "inner rtol": 1e-6,
              "inner maxiter": 1000,
              "inner monitor": False,
              "inner accel order": 0,  # >1 diverges always, 1 works with gmres only.
              "AAR order": 10,
              "AAR p": 5,
              "AAR omega": 1,
              "AAR beta": 1,
              "dsNs": dsNs,
              "dsNf": dsNf,
              "ff_vol": ff_vol,
              "fs_vol": fs_vol,
              "ff_sur": ff_sur,
              "fs_sur": fs_sur,
              "p_source": p_source}

problem = Poromechanics(parameters, mesh, parser)

# Set up BCs

V = problem.V

bcs_s = [DirichletBC(V.sub(0), Constant((0, 0)), markers, BOTTOM)]


def boundary_foot(x, on_boundary):
    return on_boundary and near(x[1], length) and abs(x[0] - length / 2) < length / 4


def boundary_foot_not(x, on_boundary):
    return on_boundary and not(near(x[1], length) and abs(x[0] - length / 2) < length / 4)


bcs_f = [DirichletBC(V.sub(1), Constant((0, 0)), boundary_foot)]
# At complement of intersection of solid and fluid
bcs_p = [DirichletBC(V.sub(2), Constant(0), boundary_foot_not)]
bcs = bcs_s + bcs_f

problem.set_bcs(bcs, bcs_p)
problem.solve()
problem.print_timings()
