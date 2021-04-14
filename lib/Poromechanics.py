from lib.AbstractPhysics import AbstractPhysics
from lib.Assembler import PoromechanicsAssembler
from lib.IndexSet import IndexSet
from lib.Solver import Solver
import dolfin as df


class Poromechanics(AbstractPhysics):
    def __init__(self, parameters, mesh, parser):
        super().__init__(parameters, mesh, parser)
        V = df.FunctionSpace(mesh,
                             df.MixedElement(df.VectorElement('CG', mesh.ufl_cell(), parameters["fe degree solid"]),
                                             df.VectorElement('CG', mesh.ufl_cell(),
                                                              parameters["fe degree fluid"]),
                                             df.FiniteElement('CG', mesh.ufl_cell(), parameters["fe degree pressure"])))
        self.V = V
        self.index_map = IndexSet(V)
        self.pprint("---- Problem dofs={}".format(V.dim()))
        self.assembler = PoromechanicsAssembler(parameters, V)
        # Start by assembling system matrices
        self.assembler.assemble()

        self.sol = df.Function(V)
        self.us_nm1, self.uf_nm1, self.p_nm1 = self.sol.split(True)
        self.us_nm2 = self.us_nm1.copy(True)

    def set_bcs(self, bcs, bcs_diff):
        """
        Set boundary conditions to both physics. Assumed to be constant.
        """
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

    def create_solver(self, A, P, P_diff, b):

        # First create preconditioner
        from lib.Preconditioner import Preconditioner
        pc = Preconditioner(self.index_map, A, P, P_diff, self.parameters, self.bcs_sub_pressure)
        pc = pc.get_pc()

        solver = Solver(A, b, pc, self.parameters, self.index_map)
        return solver.get_solver()
        # # Then create linear solver
        # solver_type = self.parameters["solver type"]
        # atol = self.parameters["solver atol"]
        # rtol = self.parameters["solver rtol"]
        # maxiter = self.parameters["solver maxiter"]
        # monitor_convergence = self.parameters["solver monitor"]
        # if solver_type == "aar":
        #     from lib.AAR import AAR
        #     order = self.parameters["AAR order"]
        #     p = self.parameters["AAR p"]
        #     omega = self.parameters["AAR omega"]
        #     beta = self.parameters["AAR beta"]
        #     return AAR(order, p, omega, beta, A.mat(), x0=None, pc=pc,
        #                atol=atol, rtol=rtol, maxiter=maxiter, monitor_convergence=monitor_convergence)
        # else:
        #     from petsc4py import PETSc
        #     solver = PETSc.KSP().create()
        #     solver.setOperators(A.mat())
        #     solver.setType(solver_type)
        #     solver.setTolerances(rtol, atol, 1e20, maxiter)
        #     solver.setPC(pc)
        #     if solver_type == "gmres":
        #         solver.setGMRESRestart(maxiter)
        #     if monitor_convergence:
        #         PETSc.Options().setValue("-ksp_monitor", None)
        #
        #     solver.setFromOptions()
        #     return solver

    def solve_time_step(self, t):
        A = self.assembler.getMatrix()
        P, P_diff = self.assembler.getPreconditioners()
        b = self.assembler.getRHS(t, self.us_nm1, self.us_nm2, self.uf_nm1, self.p_nm1)

        for bc in self.bcs:
            for obj in [A, P, P_diff, b]:
                bc.apply(obj)
        for bc in self.bcs_diff:
            bc.apply(P_diff)
        solver = self.create_solver(A, P, P_diff, b)
        solver.solve(b.vec(), self.sol.vector().vec())

        self.sol.vector().apply("")
        # Update solution
        us, uf, p = self.sol.split(True)
        df.assign(self.us_nm2, self.us_nm1)
        df.assign(self.us_nm1, us)
        df.assign(self.uf_nm1, uf)
        df.assign(self.p_nm1, p)
        return solver.getIterationNumber()
