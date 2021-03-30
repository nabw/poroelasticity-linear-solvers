from AbstractPhysics import AbstractPhysics
from Assembler import Assembler


class Poromechanics(AbstractPhysics):
    def __init__(self, parameters, mesh):
        super().__init__(self, parameters, mesh)
        V = FunctionSpace(mesh,
                          MixedElement(VectorElement('CG', mesh.ufl_cell(), parameters["fe_degree_solid"]),
                                       VectorElement('CG', mesh.ufl_cell(),
                                                     parameters["fe_degree_fluid"]),
                                       FiniteElement('CG', mesh.ufl_cell(), parameters["fe_degree_pressure"])))
        self.V = V
        self.assembler = Assembler(parameters, V)
        # Start by assembling system matrices
        assembler.assemble()

        self.sol = Function(V)
        self.us_nm1, self.uf_nm1, self.p_nm1 = self.sol.split(True)

    def set_bcs(self, bcs):
        """
        Set boundary conditions to both physics. Assumed to be constant.
        """
        self.bcs = bcs

    def create_solver(self, A, P, P_diff):

        # First create preconditioner
        from Preconditioner import Preconditioner
        pc_type = self.parameters["pc type"]
        inner_pc_type = self.parameters["inner pc type"]
        pc = Preconditioner(self.V, P, P_diff, pc_type, inner_pc_type))

        # Then create linear solver
        solver_type=self.parameters["solver type"]
        atol=self.parameters["solver atol"]
        rtol=self.parameters["solver rtol"]
        maxiter=self.parameters["solver maxiter"]
        monitor_convergence=self.parameters["solver monitor"]
        if solver_type == "AAR":
            from AAR import AAR
            order=self.parameters["aar order"]
            p=self.parameters["aar p"]
            omega=self.parameters["aar omega"]
            beta=self.parameters["aar beta"]
            return AAR(order, p, omega, beta, A.mat(), x0 = None, pc = pc,
                       atol = atol, rtol = rtol, maxiter = maxiter, monitor_convergence = monitor_convergence)
        else:
            from petsc4py import PETSc
            solver=PETSc.KSP().create()
            solver.setOperators(A.mat())
            solver.setType(solver_type)
            solver.setTolerances(atol, rtol, 1e20, maxiter)
            solver.setPC(pc)
            if solver_type == "gmres":
                solver.setGMRESRestart(maxiter)
            if monitor_convergence:
                PETSc.Options().setValue("-ksp_monitor", None)
            return solver

    def solve_time_step(self, t):
        A=self.assembler.getMatrix()
        P, P_diff=self.assembler.getPreconditioners()
        b=self.assembler.getRHS(t)

        for bc in self.bcs:
            for obj in [A, P, P_diff, b]:
                bc.apply(obj)
        solver=self.create_solver(A, P, P_diff)
        solver.solve(b.vec(), self.sol.vector().vec())

        # Update solution
        assign(self.us_nm2, self.us_nm1)
        us, uf, p=self.sol.split()
        assign(self.us_nm1, us)
        assign(self.uf_nm1, uf)
        assign(self.p_nm1, p)
