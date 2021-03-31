from lib.AbstractPhysics import AbstractPhysics
from lib.Assembler import PoromechanicsAssembler
from fenics import *


class Poromechanics(AbstractPhysics):
    def __init__(self, parameters, mesh):
        super().__init__(parameters, mesh)
        V = FunctionSpace(mesh,
                          MixedElement(VectorElement('CG', mesh.ufl_cell(), parameters["fe degree solid"]),
                                       VectorElement('CG', mesh.ufl_cell(),
                                                     parameters["fe degree fluid"]),
                                       FiniteElement('CG', mesh.ufl_cell(), parameters["fe degree pressure"])))
        self.V = V
        self.assembler = PoromechanicsAssembler(parameters, V)
        # Start by assembling system matrices
        self.assembler.assemble()

        self.sol = Function(V)
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

    def create_solver(self, A, P, P_diff):

        # First create preconditioner
        from lib.Preconditioner import Preconditioner
        pc_type = self.parameters["pc type"]
        inner_ksp_type = self.parameters["inner ksp type"]
        inner_pc_type = self.parameters["inner pc type"]
        inner_rtol = self.parameters["inner rtol"]
        inner_atol = self.parameters["inner atol"]
        inner_maxiter = self.parameters["inner maxiter"]
        inner_accel_order = self.parameters["inner accel order"]
        pc = Preconditioner(self.V, A, P, P_diff, pc_type, inner_ksp_type, inner_pc_type,
                            inner_rtol, inner_atol, inner_maxiter, inner_accel_order, self.bcs_sub_pressure)
        pc = pc.get_pc()

        # Then create linear solver
        solver_type = self.parameters["solver type"]
        atol = self.parameters["solver atol"]
        rtol = self.parameters["solver rtol"]
        maxiter = self.parameters["solver maxiter"]
        monitor_convergence = self.parameters["solver monitor"]
        if solver_type == "aar":
            from lib.AAR import AAR
            order = self.parameters["AAR order"]
            p = self.parameters["AAR p"]
            omega = self.parameters["AAR omega"]
            beta = self.parameters["AAR beta"]
            return AAR(order, p, omega, beta, A.mat(), x0=None, pc=pc,
                       atol=atol, rtol=rtol, maxiter=maxiter, monitor_convergence=monitor_convergence)
        else:
            from petsc4py import PETSc
            solver = PETSc.KSP().create()
            solver.setOperators(A.mat())
            solver.setType(solver_type)
            solver.setTolerances(rtol, atol, 1e20, maxiter)
            solver.setPC(pc)
            if solver_type == "gmres":
                solver.setGMRESRestart(maxiter)
            if monitor_convergence:
                PETSc.Options().setValue("-ksp_monitor", None)
            solver.setFromOptions()
            return solver

    def solve_time_step(self, t):
        A = self.assembler.getMatrix()
        P, P_diff = self.assembler.getPreconditioners()
        b = self.assembler.getRHS(t, self.us_nm1, self.us_nm2, self.uf_nm1, self.p_nm1)

        for bc in self.bcs:
            for obj in [A, P, P_diff, b]:
                bc.apply(obj)
        for bc in self.bcs_diff:
            bc.apply(P_diff)
        solver = self.create_solver(A, P, P_diff)
        solver.solve(b.vec(), self.sol.vector().vec())

        self.sol.vector().apply("")
        # Update solution
        assign(self.us_nm2, self.us_nm1)
        us, uf, p = self.sol.split(True)
        assign(self.us_nm1, us)
        assign(self.uf_nm1, uf)
        assign(self.p_nm1, p)
