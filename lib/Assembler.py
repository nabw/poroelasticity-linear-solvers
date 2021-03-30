from dolfin import *


class Assembler:
    """
    Class in charge of assembling the problem matrix and rhs.
    """

    def __init__(self, parameters, V):
        self.parameters = parameters

        # Geometric and discretization info
        self.V = V
        self.dim = V.mesh().geometric_dimension()
        self.dsNs = parameters.dsNs
        self.dsNf = parameters.dsNf
        self.ff_vol = parameters.ff_vol
        self.ff_sur = parameters.ff_sur
        self.fs_vol = parameters.fs_vol
        self.fs_sur = parameters.fs_sur
        self.p_source = parameters.p_source

        # Memory allocation
        self.A = PETScMatrix()
        self.P = PETScMatrix()
        self.P_diff = PETScMatrix()
        self.b = PETScVector()

        # Set parameters
        self.mu_s = Constant(parameters["mu_s"])
        self.lmbda = Constant(parameters["lmbda"])
        self.rhos = Constant(parameters["rhos"])
        self.rhof = Constant(parameters["rhof"])
        self.mu_f = Constant(parameters["mu_f"])
        self.phi0 = Constant(parameters["phi0"])
        self.ks = Constant(parameters["ks"])
        self.kf = Constant(parameters["kf"])
        self.ikf = inv(kf)
        self.dt = Constant(parameters["dt"])

        # Aux params
        self.phis = 1 - self.phi0
        self.idt = 1/self.dt

        # Stabilization
        self.betas = Constant(parameters["betas"])
        self.betaf = Constant(parameters["betaf"])
        self.betap = Constant(parameters["betap"])

    def assemble(self):

        def hooke(ten):
            return 2 * self.mu_s * ten + self.lmbda * tr(ten) * Identity(ten.ufl_shape[0])

        def eps(vec):
            return sym(grad(vec))

        us, vf, p = TrialFunctions(self.V)
        v, w, q = TestFunctions(self.V)

        # First base matrix
        a_s = (self.rhos * self.idt**2 * phis * dot(us, v)
               + inner(hooke(eps(us)), eps(v))
               - p * div(self.phis * v)
               - self.phi0 ** 2 * dot(self.ikf * (vf - self.idt * us), v)) * dx

        a_f = (self.rhof * self.idt * self.phi0 * dot(vf, w)
               + 2. * self.mu_f *
               inner(self.phi0 * eps(vf), eps(w))
               - p * div(self.phi0 * w)
               + self.phi0 ** 2 * dot(self.ikf * (vf - self.idt * us), w)) * dx

        a_p = (self.phis**2 * self.idt / self.ks * p * q
               + div(self.phi0 * vf) * q
               + div(self.phis * self.idt * us) * q) * dx

        assemble(a_s + a_f + a_p, tensor=self.A)

        # Then, preconditioner matrices (FS and DIFF)
        if prec_type == "undrained":
            N = self.ks / self.phis**2

            a_s = (self.rhos * self.idt**2 * self.phis * dot(us, v)
                   + inner(hooke(eps(us)), eps(v))
                   + N * div(self.phis * us) * div(self.phis * v)
                   - self.phi0 ** 2 * dot(self.ikf * (- self.idt * us), v)) * dx

            a_f = (self.rhof * self.idt * self.phi0 * dot(vf, w)
                   + 2. * self.mu_f * self.phi0 *
                   inner(eps(vf), eps(w))
                   - p * div(self.phi0 * w)
                   + self.phi0 ** 2 * dot(self.ikf * (vf - self.idt * us), w)) * dx

            a_p = (self.phis**2 * self.idt / self.ks * p * q
                   + div(self.phi0 * vf) * q
                   + div(self.phis * self.idt * us) * q) * dx
            a_p_diff = 0*q*dx
        elif prec_type == "diagonal-stab":
            beta_s_hat = self.betas

            a_s = (self.rhos * self.idt**2 * self.phis * dot(us, v)
                   + inner(hooke(eps(us)), eps(v))
                   - p * div(self.phis * v)
                   - self.phi0**2 * dot(self.ikf * (vf - (1. + beta_s_hat) * self.idt * us), v)) * dx

            beta_f_hat = self.betaf

            a_f = (self.rhof * self.idt * self.phi0 * dot(vf, w)
                   + 2. * self.mu_f * self.phi0 *
                   inner(eps(vf), eps(w))
                   - p * div(self.phi0 * w)
                   + (1. + beta_f_hat) * self.phi0**2 * dot(self.ikf * vf, w)) * dx

            beta_p_hat = self.betap
            beta_p = beta_p_hat * self.phis**2 / \
                (self.dt * (2. * self.mu_s / self.dim + self.lmbda))

            a_p = (self.phis**2 * self.idt / self.ks * p * q
                   + beta_p * p * q
                   + div(self.phi0 * vf) * q) * dx
            a_p_diff = 0*q*dx
        elif prec_type == "diagonal-stab-3way":
            beta_s_hat = self.betas

            a_s = (self.rhos * self.idt**2 * self.phis * dot(us, v)
                   + inner(hooke(eps(us)), eps(v))
                   - p * div(self.phis * v)
                   - self.phi0**2 * dot(self.ikf * (vf - (1. + beta_s_hat) * self.idt * us), v)) * dx

            beta_f_hat = self.betaf

            a_f = (self.rhof * self.idt * self.phi0 * dot(vf, w)
                   + 2. * self.mu_f * self.phi0 *
                   inner(eps(vf), eps(w))
                   - p * div(self.phi0 * w)
                   + (1. + beta_f_hat) * self.phi0**2 * dot(self.ikf * (vf), w)) * dx

            beta_p_hat = self.betap
            beta_p = beta_p_hat * self.phis**2 / \
                Constant(self.dt * (2. * self.mu_s / self.dim + self.lmbda))
            beta_CC1 = self.phi0 / Constant(2. * self.mu_f / self.dim)
            beta_CC2 = inv(self.rhof * self.idt / self.phi0 + self.ikf)

            a_p = (self.phis**2 * self.idt / self.ks * p * q
                   + (beta_p + beta_CC1) * p * q) * dx

            a_p_diff = (self.phis**2 * self.idt / self.ks * p * q
                        + beta_p * p * q
                        + dot(beta_CC2 * grad(p), grad(q))) * dx
        assemble(a_s + a_f + a_p, tensor=self.P)
        assemble(a_s + a_f + a_p_diff, tensor=self.P_diff)

    def getMatrix(self):
        """
        Return Dolfin::PETScMatrix, convenient for BCs.
        """
        return self.A

    def getRHS(self, t):
        """
        Get Dolfin::PETScVector RHS at time t
        """
        # Compute solid residual
        rhs_s_n = dot(self.fs_sur(t), v) * self.dsNs
        lhs_s_n = dot(self.rhos * self.idt**2 * self.phis * (-2. * us_nm1 + us_nm2), v) * \
            dx - self.phi0**2 * dot(self.ikf * (- self.idt * (- us_nm1)), w) * dx
        r_s = rhs_s_n - lhs_s_n

        # Compute fluid residual
        rhs_f_n = dot(self.ff_sur(t), w) * self.dsNf

        lhs_f = dot(self.rhof * self.idt * self.phi0 * (- uf_nm1), w) + self.phi0**2 * \
            dot(self.ikf * (-self.idt * (-us_nm1)), w)
        lhs_f_n = lhs_f * dx

        r_f = rhs_f_n - lhs_f_n

        # Compute pressure residual
        rhs_p_n = 1 / self.rhof * self.p_source(t) * q * dx

        D_sf = div(self.idt * selfphis * (- us_nm1)) * q
        M_p = self.phis**2 / Constant(self.ks * self.dt) * (- p_nm1) * q
        lhs_p_n = (M_p + D_sf) * dx
        r_p = rhs_p_n - lhs_p_n

        assemble(rhs_s_n + rhs_f_n + rhs_p_n, tensor=self.b)
        return self.b