from petsc4py import PETSc
from lib.AndersonAcceleration import AndersonAcceleration


class PreconditionerCC(object):

    def __init__(self, M, M_diff, V, flag_3_way, inner_prec_type="lu", w1=1.0, w2=0.1, accel_order=0):
        self.M = M
        self.M_diff = M_diff
        self.flag_3_way = flag_3_way
        self.w1 = w1
        self.w2 = w2
        self.ns = V.sub(0).dim()
        self.nf = V.sub(1).dim()
        self.np = V.sub(2).dim()
        self.dofmap_s = V.sub(0).dofmap().dofs()
        self.dofmap_f = V.sub(1).dofmap().dofs()
        self.dofmap_p = V.sub(2).dofmap().dofs()
        self.dofmap_fp = sorted(self.dofmap_f + self.dofmap_p)
        self.inner_prec_type = inner_prec_type
        self.anderson = AndersonAcceleration(accel_order)

    def setUp(self, pc):
        # Here we build the PC object that uses the concrete,
        # assembled matrix A.  We will use this to apply the action
        # of A^{-1}
        self.pc_s = PETSc.PC().create()
        self.pc_f = PETSc.PC().create()
        self.pc_p = PETSc.PC().create()
        self.pc_fp = PETSc.PC().create()
        self.pc_p_diff = PETSc.PC().create()

        # Create index sets for each physics
        self.is_s = PETSc.IS().createGeneral(self.dofmap_s)
        self.is_f = PETSc.IS().createGeneral(self.dofmap_f)
        self.is_p = PETSc.IS().createGeneral(self.dofmap_p)
        self.is_fp = PETSc.IS().createGeneral(self.dofmap_fp)

        # Create temp block vectors used in apply()
        self.temp_sx = PETSc.Vec().create()
        self.temp_sy = PETSc.Vec().create()
        self.temp_fx = PETSc.Vec().create()
        self.temp_fy = PETSc.Vec().create()
        self.temp_px = PETSc.Vec().create()
        self.temp_py = PETSc.Vec().create()
        self.temp_fpx = PETSc.Vec().create()
        self.temp_fpy = PETSc.Vec().create()
        self.temp_p_diffx = PETSc.Vec().create()
        self.temp_p_diffy = PETSc.Vec().create()

        # Extract sub-matrices
        self.Ms_s = self.M.createSubMatrix(self.is_s, self.is_s)
        self.Mf_s = self.M.createSubMatrix(self.is_f, self.is_s)
        self.Mf_f = self.M.createSubMatrix(self.is_f, self.is_f)
        self.Mp_s = self.M.createSubMatrix(self.is_p, self.is_s)
        self.Mp_f = self.M.createSubMatrix(self.is_p, self.is_f)
        self.Mp_p = self.M.createSubMatrix(self.is_p, self.is_p)
        self.Mfp_s = self.M.createSubMatrix(self.is_fp, self.is_s)
        self.Mfp_fp = self.M.createSubMatrix(self.is_fp, self.is_fp)
        self.Mp_diff = self.M_diff.createSubMatrix(self.is_p, self.is_p)

        self.pc_s.setType(self.inner_prec_type)
        self.pc_s.setOperators(self.Ms_s)
        self.pc_f.setType(self.inner_prec_type)
        self.pc_f.setOperators(self.Mf_f)
        self.pc_p.setType(self.inner_prec_type)
        self.pc_p.setOperators(self.Mp_p)
        self.pc_fp.setType(self.inner_prec_type)
        self.pc_fp.setOperators(self.Mfp_fp)
        self.pc_p_diff.setType(self.inner_prec_type)
        self.pc_p_diff.setOperators(self.Mp_diff)

        if self.inner_prec_type == "lu":
            factor_method = "mumps"
            self.pc_s.setFactorSolverType(factor_method)
            self.pc_f.setFactorSolverType(factor_method)
            self.pc_p.setFactorSolverType(factor_method)
            self.pc_fp.setFactorSolverType(factor_method)
            self.pc_p_diff.setFactorSolverType(factor_method)

        if self.inner_prec_type == "hypre":
            hypre_type = "boomeramg"
            self.pc_s.setHYPREType(hypre_type)
            self.pc_f.setHYPREType(hypre_type)
            self.pc_p.setHYPREType(hypre_type)
            self.pc_fp.setHYPREType(hypre_type)
            self.pc_p_diff.setHYPREType(hypre_type)

    def apply(self, pc, x, y):
        # Result is y = A^{-1}x

        x.getSubVector(self.is_s, self.temp_sx)
        y.getSubVector(self.is_s, self.temp_sy)
        self.pc_s.apply(self.temp_sx, self.temp_sy)
        # TODO: use mmult to avoid creating temp vectors for off-diagonal contributions
        if self.flag_3_way:
            x.getSubVector(self.is_f, self.temp_fx)
            y.getSubVector(self.is_f, self.temp_fy)
            x.getSubVector(self.is_p, self.temp_px)
            y.getSubVector(self.is_p, self.temp_py)
            x.getSubVector(self.is_p, self.temp_p_diffx)
            y.getSubVector(self.is_p, self.temp_p_diffy)
            self.pc_f.apply(self.temp_fx - self.Mf_s * self.temp_sy, self.temp_fy)
            self.pc_p.apply(self.temp_px - self.Mp_s * self.temp_sy -
                            self.Mp_f * self.temp_fy, self.temp_py)

            # Same rhs as for pc_p
            self.temp_p_diffx = self.temp_px - self.Mp_s * self.temp_sy - self.Mp_f * self.temp_fy
            # Apply bc to pressure rhs first
            n_bcs = len(bcs_sub_pressure)
            self.temp_p_diffx.setValues(bcs_sub_pressure, np.zeros(n_bcs))
            self.pc_p_diff.apply(self.temp_p_diffx, self.temp_p_diffy)

            self.temp_py.scale(self.w1)
            self.temp_py.axpy(self.w2, self.temp_p_diffy)
            # with open('out.txt', 'a') as f:
            # f.write("{}\n".format(self.temp_p_diffy.norm()))

            x.restoreSubVector(self.is_f, self.temp_fx)
            y.restoreSubVector(self.is_p, self.temp_py)
            x.restoreSubVector(self.is_p, self.temp_px)
            y.restoreSubVector(self.is_f, self.temp_fy)
        else:  # use 2way
            x.getSubVector(self.is_fp, self.temp_fpx)
            y.getSubVector(self.is_fp, self.temp_fpy)

            # compute A_fp_s ys, ys resulting vector from before
            self.pc_fp.apply(self.temp_fpx - self.Mfp_s * self.temp_sy, self.temp_fpy)
            x.restoreSubVector(self.is_fp, self.temp_fpx)
            y.restoreSubVector(self.is_fp, self.temp_fpy)

        x.restoreSubVector(self.is_s, self.temp_sx)
        y.restoreSubVector(self.is_s, self.temp_sy)

        self.anderson.get_next_vector(y)


class Preconditioner:
    def __init__(self, V, A, P, P_diff, pc_type, inner_prec_type, inner_accel_order):
        self.V = V
        self.A = A
        self.P = P
        self.P_diff = P_diff
        self.pc_type = pc_type
        self.inner_prec_type = inner_prec_type
        self.inner_accel_order = inner_accel_order
        if pc_type not in ("undrained", "diagonal", "diagonal 3-way"):
            import sys
            sys.exit("pc type must be one of undrained, diagonal, diagonal 3-way.")

    def get_pc(self):
        flag_3_way = self.pc_type == "diagonal 3-way"
        ctx = PreconditionerCC(self.P.mat(), self.P_diff.mat(), self.V,
                               flag_3_way, self.inner_prec_type, 1.0, 0.1, self.inner_accel_order)
        pc = PETSc.PC().create()
        pc.setType('python')
        pc.setPythonContext(ctx)
        pc.setOperators(self.A.mat())
        pc.setUp()
        return pc
