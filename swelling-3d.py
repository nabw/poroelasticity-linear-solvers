from petsc4py import PETSc
from dolfin import *
from lib.MeshCreation import generate_cube, generateBoundaryMeasure, NONE
from lib.Poromechanics import Poromechanics
from lib.Parser import Parser
from time import time
parameters["mesh_partitioner"] = "ParMETIS"

initial_time = time()
parser = Parser()
Nelements = 6
if parser.options.N:
    Nelements = parser.options.N
side_length = 1e-2
mesh, markers, XP, XM, YP, YM, ZP, ZM = generate_cube(
    Nelements, side_length)


neumann_solid_markers = [XP, YP, ZP]
neumann_fluid_markers = [XM, YM]

dsNs = generateBoundaryMeasure(mesh, markers, neumann_solid_markers)
dsNf = generateBoundaryMeasure(mesh, markers, neumann_fluid_markers)

# Set up load terms
fs_vol = ff_vol = fs_sur = lambda t: Constant((0., 0., 0.))


def p_source(t): return Constant(0.0)


def ff_sur(t):
    return Constant(-1e3 * 0.1 * (1 - exp(-(t**2) / 0.25))) * FacetNormal(mesh)


parameters = {"mu_f": 0.035,
              "rhof": 1e3,
              "rhos": 1e3,
              "phi0": 0.1,
              "mu_s": 4000,
              "lmbda": 700,
              "ks": 1e5,
              "kf": 1e-7,
              "dt": 0.1,
              "t0": 0.0,
              "tf": 0.1,
              "fe degree solid": 2,
              "fe degree fluid": 2,
              "fe degree pressure": 1,
              "maxiter": 1000,
              "output solutions": True,
              # "output_name": "monolithic",
              "output name": "swelling-3d",
              "betas": -0.5,
              "betaf": 0.,
              "betap": 1.,
              "solver atol": 1e-8,
              "solver rtol": 1e-6,
              "solver maxiter": 500,
              "solver monitor": False,
              "solver type": "gmres",  # cg, gmres, aar
              "pc type": "diagonal 3-way",  # diagonal, undrained, diagonal 3-way
              "inner ksp type": "cg",  # preonly, gmres, cg, bicgstab,
              "inner pc type": "hypre",  # bjacobi, ilu, hypre, lu, gamg, asm
              "inner rtol": 0,
              "inner atol": 1e-6,
              "inner maxiter": 100,
              "inner monitor": True,
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

bcs_s = [DirichletBC(V.sub(0).sub(0), Constant(0), markers, XM),
         DirichletBC(V.sub(0).sub(1), Constant(0), markers, YM),
         DirichletBC(V.sub(0).sub(2), Constant(0), markers, ZM)]

bcs_f = [DirichletBC(V.sub(1), Constant((0, 0, 0)), markers, ZM),
         DirichletBC(V.sub(1), Constant((0, 0, 0)), markers, ZP)]
# Intersetion is ZM, so bc for pressure is set on the complement.
bcs_p = [DirichletBC(V.sub(2), Constant(0), markers, XM),
         DirichletBC(V.sub(2), Constant(0), markers, XP),
         DirichletBC(V.sub(2), Constant(0), markers, YM),
         DirichletBC(V.sub(2), Constant(0), markers, YP),
         DirichletBC(V.sub(2), Constant(0), markers, ZP)]
bcs = bcs_s + bcs_f

problem.set_bcs(bcs, bcs_p)
problem.solve()
