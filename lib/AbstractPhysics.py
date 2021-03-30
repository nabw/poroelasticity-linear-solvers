
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
        required_fields = ["t0", "tf", "dt", "output_name"]
        assert all([x in parameters.keys() for x in required_fields]
                   ), "Missing arguments in parameters: {}".format(required_fields)

        # Time parameters
        self.t0 = parameters["t0"]
        self.t = self.t0
        self.tf = parameters["tf"]
        self.dt = parameters["dt"]

        # Export
        self.xdmf = XDMFFile(
            "output/{}.xdmf".format(parameters["output_name"]))
        self.xdmf.parameters["functions_share_mesh"] = True
        self.xdmf.parameters["flush_output"] = True
        self.xdmf.parameters["rewrite_function_mesh"] = False

        self.us_nm1 = None
        self.uf_nm1 = None
        self.p_nm1 = None

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

    def solve(self, f_vol_solid, f_sur_solid, f_vol_fluid, f_sur_fluid, p_source=None):
        """
        Solve poromechanics problem. Problem loads are assumed to give a function when evaluated at a specific time. For example:
        f_vol = lambda t: Constant((1,1))
        """

        # All functions start as 0 (for now), so no modifications are required.

        from time import time
        current_time = time()
        print("Begining simulation")
        iterations = []

        if self.export_solutions:
            self.export(0)

        if p_source:
            pass
        else:
            def p_source(t): return Constant(0)

        while self.t < self.tf:

            self.t += self.dt
            iter_count = self.solve_time_step(f_vol_solid(
                self.t), f_sur_solid(self.t), f_vol_fluid(self.t), f_sur_fluid(self.t), p_source(self.t))
            iterations.append(iter_count)
            print("-- Solved time t={:.4f} in {:.3f}s and {} iterations".format(self.t,
                                                                                time() - current_time, iter_count))
            if self.export_solutions:
                self.export(self.t)
            current_time = time()

        self.iterations = iterations
        self.avg_iter = self.compute_average_iterations()
