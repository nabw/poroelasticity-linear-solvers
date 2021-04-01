from optparse import OptionParser


class Parser:
    def __init__(self):
        """
        Read some input parameters and generate them as a dicitonary which is compatible
        with the one used for the physics. This should be then used to update the parameters
        as: params.update(out), which replaces all of the given entries.
        """
        parser = OptionParser(add_help_option=False)
        parser.add_option("-h", "--help", action="help")
        # Add real options
        parser.add_option("-N", type="int", dest="N", help="Number of elements per side")
        parser.add_option("-solver-type", type="str", dest="solver_type",
                          help="Type of linear solver: gmres, cg, aar")
        parser.add_option("-pc-type", type="str", dest="pc_type",
                          help="Type of splitting preconditioner: diagonal, undrained, diagonal 3-way")
        parser.add_option("-fe-solid", type="int", dest="fe_s",
                          help="Finite element degree of solid")
        parser.add_option("-monitor", action="store_true", dest="monitor",
                          help="Monitor linear solver convergence")
        options, _ = parser.parse_args()

        options_dict = {}  # Empty dictionary
        if options.N:
            options_dict["N"] = options.N
        if options.solver_type:
            options_dict["solver type"] = options.solver_type
        if options.pc_type:
            options_dict["pc type"] = options.pc_type
        if options.fe_s:
            options_dict["fe degree solid"] = options.fe_s
        if options.monitor:
            options_dict["solver monitor"] = True
        self.options_dict = options_dict
        self.options = options
