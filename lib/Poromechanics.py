from lib.AbstractPhysics import AbstractPhysics
from lib.Assembler import PoromechanicsAssembler
from lib.IndexSet import IndexSet
from lib.Solver import Solver
from lib.Printing import parprint
from mpi4py import MPI
from time import perf_counter as time
import dolfin as df


class Poromechanics(AbstractPhysics):
    def __init__(self, parameters, mesh, parser):
        super().__init__(parameters, mesh, parser)
        V = df.FunctionSpace(self.mesh,
                             df.MixedElement(df.VectorElement('CG', self.mesh.ufl_cell(), parameters["fe degree solid"]),
                                             df.VectorElement('CG', self.mesh.ufl_cell(),
                                                              parameters["fe degree fluid"]),
                                             df.FiniteElement('CG', self.mesh.ufl_cell(), parameters["fe degree pressure"])))
        self.V = V
        self.two_way = True
        self.three_way = False
        if "3-way" in self.parameters["pc type"]:
            self.two_way = False
            self.three_way = True

        parprint("---- Problem dofs={}, h={}, solving with {} procs".format(V.dim(), mesh.hmin(), MPI.COMM_WORLD.size))
        self.assembler = PoromechanicsAssembler(parameters, V, self.three_way)

        self.index_map = IndexSet(V, self.two_way)
        # Start by assembling system matrices
        self.assembler.assemble()

        self.sol = df.Function(V)
        self.us_nm1, self.uf_nm1, self.p_nm1 = self.sol.split(True)
        self.us_nm2 = self.us_nm1.copy(True)

        self.first_timestep = True

    def set_bcs(self, bcs, bcs_diff):
        """
        Set boundary conditions to both physics. Assumed to be constant.
        """
        t0 = time()
        self.bcs = bcs
        self.bcs_diff = bcs_diff

        # Create map for pressure dofs, used in 3way CC preconditioner
        dofs_p = self.V.sub(2).dofmap().dofs()
        bcs_sub_pressure = []
        for b in bcs_diff:
            bc_vals = b.get_boundary_values().keys()
            for i, dof in enumerate(dofs_p):
                if dof in bc_vals:
                    bcs_sub_pressure.append(i)
        self.bcs_sub_pressure = bcs_sub_pressure
        parprint("---- [BC] Created inverse pressure BC in {:.3f}s".format(time() - t0))

    def create_solver(self, A, P, P_diff, b):

        # First create preconditioner
        from lib.Preconditioner import Preconditioner
        self.pc = Preconditioner(self.index_map, A, P, P_diff,
                                 self.parameters, self.bcs_sub_pressure)
        pc = self.pc.get_pc()

        solver = Solver(A, b, pc, self.parameters, self.index_map)
        solver.create_solver(A, b, pc)
        self.solver = solver

    def solve_time_step(self, t):

        A = self.assembler.getMatrix()
        P, P_diff = self.assembler.getPreconditioners()
        b = self.assembler.getRHS(t, self.us_nm1, self.us_nm2, self.uf_nm1, self.p_nm1)

        for bc in self.bcs:
            for obj in [A, P, b]:
                bc.apply(obj)
            if self.three_way:
                bc.apply(P_diff)
        if self.three_way:
            for bc in self.bcs_diff:
                bc.apply(P_diff)
        if self.first_timestep:
            self.create_solver(A, P, P_diff, b)
            self.first_timestep = False

        self.solver.set_up()
        self.solver.solve(b.vec(), self.sol.vector().vec())

        self.sol.vector().apply("")  # Update ghost dofs
        # Update solution
        us, uf, p = self.sol.split(True)
        df.assign(self.us_nm2, self.us_nm1)
        df.assign(self.us_nm1, us)
        df.assign(self.uf_nm1, uf)
        df.assign(self.p_nm1, p)
        return self.solver.getIterationNumber()

    def print_timings(self):
        self.pc.print_timings()
        self.solver.print_timings()
