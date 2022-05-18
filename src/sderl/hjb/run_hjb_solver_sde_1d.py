from sderl.hjb.hjb_solver_1d import SolverHJB1d
import molecules.models.double_well as dw

def main():

    # set environment 
    d = 1
    alpha_i = 1.
    beta = 2.
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])

    # initialize hjb solver
    sol_hjb = SolverHJB1d(env, h=0.01, lb=-3., ub=3.)

    # compute soltuion
    if not sol_hjb.load():

        # compute hjb solution 
        sol_hjb.solve_bvp()
        sol_hjb.compute_value_function()
        sol_hjb.compute_optimal_control()

        sol_hjb.save()

    # report solution
    sol_hjb.write_report(x=-1.)

    # evaluate in grid
    sol_hjb.get_perturbed_potential_and_drift()

    # plots
    sol_hjb.plot_psi()
    sol_hjb.plot_value_function()
    sol_hjb.plot_perturbed_potential()
    sol_hjb.plot_control()
    sol_hjb.plot_perturbed_drift()


if __name__ == "__main__":
    main()
