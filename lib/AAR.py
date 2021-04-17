import numpy as np
from petsc4py import PETSc
from mpi4py import MPI


class AAR:

    def __init__(self, order, p, omega, beta, matA, x0=None, pc=None, atol=1e-12, rtol=1e-8, maxiter=1000, monitor_convergence=False):
        self.order = order
        self.x0 = x0
        self.p = p
        self.omega = omega
        self.beta = beta
        self.matA = matA
        self.atol = atol
        self.rtol = rtol
        self.maxiter = maxiter
        self.monitor_convergence = monitor_convergence
        self.F = []
        self.X = []
        self.F0 = []  # For global vectors
        self.xk = None
        self.fk = None
        self.delta_xk = None
        self.delta_fk = None
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.d_fk0 = None
        self.fk0 = None

        # If no PC given, use an ILU
        if not pc:
            pc = self.solver.getPC()
            pc.setType('ilu')
            pc.setOperators(matA)
            pc.setFactorLevels(3)
            self.pc = pc
        else:
            self.pc = pc

    def set_up(self):
        # TODO: Set convergence test
        pass

    def solve(self, b, sol):

        if not self.x0:
            self.x0 = b.copy()
            self.x0.zeroEntries()

        # Initialize current vectors
        self.temp_vec = None
        self.xk = self.x0.copy()
        self.fk = b.copy()
        self.fk.axpy(-1, self.matA * self.x0)
        self.delta_xk = b.copy()
        self.delta_fk = b.copy()
        self.temp_vec = b.copy()
        # Init global vectors and scatterer
        self.scatter, aux = PETSc.Scatter.toZero(b)
        self.d_fk0 = aux.copy()
        self.d_fk0.zeroEntries()
        self.fk0 = aux.copy()
        self.fk0.zeroEntries()

        error0 = self.fk.norm()
        err_abs = error0
        err_rel = 1
        it = 0
        alpha = None
        current_type = ""
        while err_abs > self.atol and err_rel > self.rtol and it < self.maxiter:

            self.fk.copy(self.delta_fk)
            self.xk.copy(self.delta_xk)
            self.update_residual(b)  # Update fk value
            self.delta_fk.aypx(-1, self.fk)

            self.F.append(self.delta_fk.copy())
            if len(self.F) > self.order:
                self.F.pop(0)

            # Update global vectors on first cpu
            self.scatter.scatter(self.delta_fk, self.d_fk0)
            self.scatter.scatter(self.fk, self.fk0)
            self.F0.append(self.d_fk0.copy())
            if len(self.F0) > self.order:
                self.F0.pop(0)

            if self.fk.norm() < 1e-14:
                pass
            # If not a natural number or first iteration
            elif it == 0 or self.order == 0 or (it + 1) / self.p % 1 > 0:
                current_type = "R"
                self.xk.axpy(self.omega, self.fk)
            else:
                current_type = "A"
                mk = min(self.order, it)
                # Process only on first core, then scatter alpha
                if self.rank == 0:
                    F = np.vstack(self.F0).T
                    Q, R = np.linalg.qr(F)
                    rhs = self.fk0
                    alpha = np.linalg.solve(R, -Q.T @ rhs)
                else:
                    alpha = None
                alpha = self.comm.bcast(alpha, root=0)
                self.xk.axpy(self.beta, self.fk)
                for i in range(mk):
                    self.xk.axpy(alpha[i], self.X[i] + self.beta * self.F[i])

            self.delta_xk.aypx(-1, self.xk)
            self.X.append(self.delta_xk.copy())
            if len(self.X) > self.order:
                self.X.pop(0)
            err_abs = self.fk.norm()
            err_rel = err_abs / error0
            it += 1

            if self.monitor_convergence and self.rank == 0:
                print("---- Iteration [{}] {:3}\tabs={:1.2e}\trel={:1.2e}".format(
                    current_type, it, err_abs, err_rel), flush=True)

        # Update solution and return number of iterations
        self.xk.copy(sol)
        self.it = it
        return it

    def getIterationNumber(self):
        return self.it

    def update_residual(self, b):
        b.copy(self.temp_vec)
        self.temp_vec.axpy(-1, self.matA * self.xk)
        # self.pc.apply(self.temp_vec, self.fk)
        self.pc.apply(self.temp_vec, self.fk)
