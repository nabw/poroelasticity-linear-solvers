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

    del solve_time_step(self, t)
