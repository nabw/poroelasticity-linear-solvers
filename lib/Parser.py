from optparse import OptionParser
from petsc4py import PETSc


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
        parser.add_option("-N", "--Nelements", type="int", dest="N",
                          help="Number of elements per side")
        parser.add_option("--N-refinements", type="int", dest="refinements",
                          help="Number of mesh refinements")
        parser.add_option("--solver-type", type="str", dest="solver_type",
                          help="Type of linear solver: gmres, cg, aar")
        parser.add_option("--pc-type", type="str", dest="pc_type",
                          help="Type of splitting preconditioner: diagonal, undrained, diagonal 3-way, undrained 3-way")
        parser.add_option("--fe-solid", type="int", dest="fe_s",
                          help="Finite element degree of solid")
        parser.add_option("--monitor", action="store_true", dest="monitor",
                          help="Monitor linear solver convergence")
        parser.add_option("--inner-monitor", action="store_true", dest="inner_monitor",
                          help="Monitor convergence of preconditioner solvers")
        parser.add_option("--inner-accel-order", type="int", dest="inner_accel_order",
                          help="Order of inner Anderson acceleration")
        parser.add_option("--output", action="store_true", dest="output",
                          help="Use this to activate solution export")
        parser.add_option("--time-final", type="float", dest="tf",
                          help="Time to end simulation")
        parser.add_option("--petsc-options", type="str", dest="options_file",
                          help="PETSc options file")

        options, _ = parser.parse_args()

        options_dict = {}  # Empty dictionary
        if options.N:
            options_dict["N"] = options.N
        if options.refinements:
            options_dict["mesh refinements"] = options.refinements
        if options.solver_type:
            options_dict["solver type"] = options.solver_type
        if options.pc_type:
            options_dict["pc type"] = options.pc_type
        if options.fe_s:
            options_dict["fe degree solid"] = options.fe_s
        if options.monitor:
            options_dict["solver monitor"] = True
        if options.inner_monitor:
            options_dict["inner monitor"] = True
        if options.inner_accel_order:
            options_dict["inner accel order"] = options.inner_accel_order
        if options.output:
            options_dict["output solutions"] = True
        if options.tf:
            options_dict["tf"] = options.tf
        if options.options_file:
            with open(options.options_file, "r") as opts_file:
                lines = opts_file.readlines()
                for _line in lines:
                    line = _line.rstrip().lstrip()  # Remove trailing and leading whitespace
                    if "#" in line or len(line) == 0:  # Skip comment lines
                        continue
                    split = line.split(" ")
                    if len(split) > 1:
                        key, val = split[0], split[-1]  # In case there is more than one space
                        PETSc.Options().setValue(key, val)
                    else:
                        PETSc.Options().setValue(line, None)
        self.options_dict = options_dict
        self.options = options
