from fenics import *
from mpi4py import MPI


class AbstractPhysics:
    """
    Abstract class in charge of solving a time-dependent physics problem.
    """

    def __init__(self, parameters, mesh):
        """
        Constructor
        """
        self.parameters = parameters
        self.mesh = mesh
        self.dim = mesh.topology().dim()

        # First assert that all required variables are in the parameters dictionary
        required_fields = ["t0", "tf", "dt", "output name"]
        assert all([x in parameters.keys() for x in required_fields]
                   ), "Missing arguments in parameters: {}".format(required_fields)

        # Time parameters
        self.t0 = parameters["t0"]
        self.t = self.t0
        self.tf = parameters["tf"]
        self.dt = parameters["dt"]

        # Export
        self.output_solutions = parameters["output solutions"]
        self.xdmf = XDMFFile(
            "output/{}.xdmf".format(parameters["output name"]))
        self.xdmf.parameters["functions_share_mesh"] = True
        self.xdmf.parameters["flush_output"] = True
        self.xdmf.parameters["rewrite_function_mesh"] = False

        self.us_nm1 = None
        self.uf_nm1 = None
        self.p_nm1 = None

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def pprint(self, *args):
        if self.rank == 0:
            print(*args)

    def export(self, time):
        """
        Export solutions, assumed to be independent (coming from collapsed spaces).
        """

        self.us_nm1.rename("displacement", "displacement")
        self.uf_nm1.rename("fluid velocity", "fluid velocity")
        self.p_nm1.rename("pressure", "pressure")
        self.xdmf.write(self.us_nm1, time)
        self.xdmf.write(self.uf_nm1, time)
        self.xdmf.write(self.p_nm1, time)

    def solve(self):
        """
        Solve poromechanics problem. Problem loads are assumed to give a function when evaluated at a specific time. For example:
        f_vol = lambda t: Constant((1,1))
        """

        # All functions start as 0 (for now), so no modifications are required.

        from time import time
        current_time = time()
        self.pprint("Begining simulation")
        iterations = []

        if self.output_solutions:
            self.export(self.t0)

        while self.t < self.tf:

            self.t += self.dt
            self.solve_time_step(self.t)
            self.pprint("-- Solved time t={:.4f} in {:.3f}s".format(self.t, time() - current_time))
            if self.output_solutions:
                self.export(self.t)
            current_time = time()
