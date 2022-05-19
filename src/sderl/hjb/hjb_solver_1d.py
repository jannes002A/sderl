import os

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import molecules.models.double_well as dw
from sderl.utils.config import DATA_DIR_PATH

class SolverHJB1d(object):
    """ This class provides a solver of the following 1d BVP by using a
        finite differences method:
            0 = LΨ − f Ψ in S
            Ψ = exp(− g) in ∂S,
        where f = 1, g = 1 and L is the infinitessimal generator
        of the not controlled 1d overdamped langevin process:
            L = - dV/dx d/dx + 1/2 d^2/dx^2

    Attributes
    ----------
    env : object
        environment object
    psi: array
        solution of the BVP problem
    value_f: array
        value function of the HJB equation
    u_opt: array
        optimal control of the HJB equation


    Methods
    -------
    __init__(sde, h)

    get_x(k)

    get_index(x)

    solve_bvp()

    compute_value_function()

    compute_optimal_control()

    save()

    load()

    write_report(x)

    plot_psi()

    plot_value_function()

    plot_perturbed_potential()

    plot_control()

    plot_perturbed_drift()

    """

    def __init__(self, env, h, lb, ub):
        """ init method

        Parameters
        ----------
        env : object
            environment object
        h: float
            step size
        lb: float
            domain lower bound
        ub: float
            domain upper bound

        Raises
        ------
        NotImplementedError
            If dimension d is greater than 1
        """

        if env.dim[0] != 1:
            raise NotImplementedError('d > 1 not supported')

        # environment
        self.env = env

        # discretization step
        self.h = h

        # discretized domain
        self.lb = lb
        self.ub = ub
        self.domain_h = jnp.arange(lb, ub+h, h, dtype=jnp.float32)

        # number of nodes
        self.n_nodes = self.domain_h.shape[0]

        # target set
        self.target_set_lb = env.stop.item()
        self.target_set_ub = ub

        # directory path
        self.dir_path = os.path.join(DATA_DIR_PATH, 'hjb-pde')

        # file path
        self.file_path = os.path.join(
            self.dir_path,
            self.env.name + str('_{:.0e}.npz'.format(self.h)),
        )


    def get_x(self, n):
        """ returns the x-coordinate of the node n

        Parameters
        ----------
        n: int
            index of the node

        Returns
        -------
        float
            point in the domain
        """
        assert n in jnp.arange(self.n_nodes), ''

        return self.domain_h[n]

    def get_index(self, x):
        """ returns the index of the point of the grid closest to x. Assumes the domain is
            an hypercube.

        Parameters
        ----------
        x: float
            point in the domain

        Returns
        -------
        idx: int
            index
        """
        idx = jnp.floor(
            (jnp.clip(x, self.lb, self.ub - 2 * self.h) + self.ub) / self.h
        ).astype(int)

        return idx

    def solve_bvp(self):
        """ solve bvp by using finite difference
        """
        # assemble linear system of equations: A \Psi = b.
        A = jnp.zeros((self.n_nodes, self.n_nodes))
        b = jnp.zeros(self.n_nodes)

        # nodes in boundary
        idx_boundary = jnp.array([0, self.n_nodes - 1])

        # nodes in target set
        idx_ts = jnp.where(
            (self.domain_h >= self.target_set_lb) & (self.domain_h <= self.target_set_ub)
        )[0]

        for n in jnp.arange(self.n_nodes):

            # assemble matrix A and vector b on S
            if n not in idx_ts and n not in idx_boundary:
                x = self.get_x(n)
                dV = self.env.grad(x).item()
                A = A.at[n, n].set(- self.env.sigma**2 / self.h**2 - 1.)
                A = A.at[n, n - 1].set(self.env.sigma**2 / (2 * self.h**2) + dV / (2 * self.h))
                A = A.at[n, n + 1].set(self.env.sigma**2 / (2 * self.h**2) - dV / (2 * self.h))
                b = b.at[n].set(0)

            # impose condition on ∂S
            elif n in idx_ts:
                x = self.get_x(n)
                A = A.at[n, n].set(1)
                b = b.at[n].set(1)

        # stability condition on the boundary: Psi should be flat

        # Psi_0 = Psi_1
        A = A.at[0, 0].set(1)
        A = A.at[0, 1].set(-1)
        b = b.at[0].set(0)

        # psi_{Nh-1} = Psi_N)
        A = A.at[-1, -1].set(1)
        A = A.at[-1, -2].set(-1)
        b = b.at[-1].set(0)

        # solve linear system and save
        psi = jnp.linalg.solve(A, b)
        self.psi = psi.reshape(self.n_nodes)
        self.solved = True

    def compute_value_function(self):
        """ this methos computes the value function
                value_f = - log (psi)
        """
        assert hasattr(self, 'psi'), ''
        self.value_function =  - jnp.log(self.psi)

    def compute_optimal_control(self):
        """ this method computes by finite differences the optimal control
                u_opt = - sigma ∇_x value_f
        """
        assert hasattr(self, 'value_function'), ''

        self.u_opt = jnp.zeros(self.n_nodes)

        # central difference approximation
        # for any k in {1, ..., Nh-2}
        # u_opt(x_k) = - sigma (Phi_{k+1} - Phi_{k-1}) / 2h

        self.u_opt = self.u_opt.at[1:-1].set(
            - self.env.sigma \
            * (self.value_function[2:] - self.value_function[:-2]) \
            / (2 * self.h)
        )
        self.u_opt = self.u_opt.at[0].set(self.u_opt[1])
        self.u_opt = self.u_opt.at[-1].set(self.u_opt[-2])

    def save(self):
        """ saves some attributes as arrays into a .npz file
        """

        # create directoreis of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # save arrays in a npz file
        jnp.savez(
            self.file_path,
            domain_h=self.domain_h,
            psi=self.psi,
            value_function=self.value_function,
            u_opt=self.u_opt,
        )

    def load(self):
        """ loads the saved arrays and sets them as attributes back
        """
        try:
            data = jnp.load(self.file_path, allow_pickle=True)

            for attr_name in data.files:

                attr = data[attr_name]
                setattr(self, attr_name, attr)

            return True

        except:
            print('no hjb-solution found with h={:.0e}'.format(self.h))
            return False

    def get_perturbed_potential_and_drift(self):
        """ computes the potential, bias potential, controlled potential, gradient,
            controlled drift
        """

        # flatten domain_h
        x = self.domain_h.reshape(self.n_nodes, self.env.dim[0])

        # potential, bias potential and tilted potential
        V = self.env.potential_batch(x).reshape(self.n_nodes)
        self.bias_potential = self.value_function * self.env.sigma**2
        self.perturbed_potential = V + self.bias_potential

        # gradient and tilted drift
        dV = self.env.grad_batch(x).reshape(self.n_nodes)
        self.perturbed_drift = - dV + self.env.sigma * self.u_opt


    def write_report(self, x):
        """ writes the hjb solver parameters

        Parameters
        ----------
        x: float
            point in the domain

        """

        # space discretization
        print('\n space discretization\n')
        print('h = {:2.4f}'.format(self.h))
        print('n of nodes = {:d}'.format(self.n_nodes))

        # psi, value function and control
        print('\n psi, value function and optimal control at x\n')

        # get index of x
        idx = self.get_index(x)

        psi = self.psi[idx]
        value_f = self.value_function[idx]
        u_opt = self.u_opt[idx]

        print('x = {:2.1f}'.format(x))
        print('psi(x) = {:2.4e}'.format(psi))
        print('value_f(x) = {:2.4e}'.format(value_f))
        print('u_opt(x) = {:2.4e}'.format(u_opt))

        # maximum value of the control
        print('\n maximum value of the optimal control\n')

        # get idx of the maximum of the control
        idx_u_max = jnp.argmax(u_opt)
        x_u_max = self.domain_h[idx_u_max]
        u_opt_max = self.u_opt[idx_u_max]

        print('x (max u_opt) = {:2.1f}'.format(x_u_max))
        print('max u_opt (x) = {:2.4e}'.format(u_opt_max))


    def plot_psi(self):
        fig, ax = plt.subplots()
        x = np.array(self.domain_h)
        y = np.array(self.psi)
        ax.set_title('$\Psi$')
        ax.set_xlabel('x')
        ax.set_xlim(-2, 2)
        ax.plot(x, y)
        plt.show()

    def plot_value_function(self):
        fig, ax = plt.subplots()
        x = np.array(self.domain_h)
        y = np.array(self.value_function)
        ax.set_title('$\Phi$')
        ax.set_xlabel('x')
        ax.set_xlim(-2, 2)
        ax.plot(x, y)
        plt.show()

    def plot_perturbed_potential(self):
        fig, ax = plt.subplots()
        x = np.array(self.domain_h)
        y = np.array(self.perturbed_potential)
        ax.set_title('$\widetilde{V}$')
        ax.set_xlabel('x')
        ax.set_xlim(-2, 2)
        ax.plot(x, y)
        plt.show()

    def plot_control(self):
        fig, ax = plt.subplots()
        x = np.array(self.domain_h)
        y = np.array(self.u_opt)
        ax.set_title('$u^*$')
        ax.set_xlabel('x')
        ax.set_xlim(-2, 2)
        ax.plot(x, y)
        plt.show()

    def plot_perturbed_drift(self):
        fig, ax = plt.subplots()
        x = np.array(self.domain_h)
        y = np.array(self.perturbed_drift)
        #ax.set_title('$$')
        ax.set_xlabel('x')
        ax.set_xlabel('x')
        ax.set_xlim(-2, 2)
        ax.plot(x, y)
        plt.show()
