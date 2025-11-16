import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy import constants

# Physical constants (atomic units)
Eh = constants.physical_constants['atomic unit of energy'][0]
kB = constants.Boltzmann / Eh

V0_default = 0.425 * constants.eV / Eh
a_default = 0.734
alpha_default = 2.0

m_mass = 1060.0
hbar = 1.0

# Asymmetric Eckart Barrier
def V_Asym_local(x, V_0=V0_default, a=a_default, alpha=alpha_default):
    """
    Local (single-bead) potential value at coordinate x.
    Works for scalar or numpy array x.
    """
    x = np.asarray(x)
    return (V_0 * (1 - alpha) / (1 + np.exp(-2 * x / a))
            + V_0 * (1 + np.sqrt(alpha))**2 / (4 * np.cosh(x / a)**2))

def V_Asym(x, V_0=V0_default, a=a_default, alpha=alpha_default):
    """
    Total potential for a ring polymer configuration x (sum over beads).
    x can be a 1D numpy array.
    """
    x = np.asarray(x)
    return np.sum(V_Asym_local(x, V_0=V_0, a=a, alpha=alpha))


# Numerical gradient and Hessian
def gradient(f, x, h=1e-5):
    n = x.size
    g = np.zeros(n)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = h
        g[i] = (f(x + dx) - f(x - dx)) / (2.0 * h)
    return g

def hessian(f, x, h=1e-4):
    n = x.size
    H = np.zeros((n, n))
    f0 = f(x)

    # Diagonal elements
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = h
        fp = f(x + dx)
        fm = f(x - dx)
        H[i, i] = (fp - 2.0 * f0 + fm) / (h**2)

    # Off-diagonal elements
    for i in range(n):
        for j in range(i + 1, n):
            dx_i = np.zeros(n)
            dx_j = np.zeros(n)
            dx_i[i] = h
            dx_j[j] = h

            fpp = f(x + dx_i + dx_j)
            fpm = f(x + dx_i - dx_j)
            fmp = f(x - dx_i + dx_j)
            fmm = f(x - dx_i - dx_j)

            val = (fpp - fpm - fmp + fmm) / (4.0 * h * h)
            H[i, j] = H[j, i] = val

    return H


# Internal spring contributions
def grad_int(Q, k_eff):
    Q_fow = np.roll(Q, -1)
    Q_back = np.roll(Q, 1)
    return k_eff * (2 * Q - Q_fow - Q_back)

def hess_int(n, k_eff):
    Hess = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    Hess[-1, 0] = -1
    Hess[0, -1] = -1
    return Hess * k_eff

def grad_polymer(Q, k_eff, V_ext, h):
    grad_ex = gradient(V_ext, Q, h)
    grad_in = grad_int(Q, k_eff)
    return grad_ex + grad_in

def Hess_polymer(Q, k_eff, V_ext, h=1e-5):
    n = Q.size
    Hess_ex = hessian(V_ext, Q, h)
    Hess_in = hess_int(n, k_eff)
    return Hess_ex + Hess_in


# Cerjan–Miller step for arbitrary index
def Polymer_step(Q, k_eff, V_ext, idx_order, fd_step=1e-4, max_step=0.5):
    Q = np.asarray(Q, dtype=float)
    g = grad_polymer(Q, k_eff, V_ext, fd_step)
    H = Hess_polymer(Q, k_eff, V_ext, fd_step)

    eigvals, eigvecs = np.linalg.eigh(H)
    k = eigvals
    U = eigvecs

    d = U.T @ g

    Hess_idx = sum(1 for i in k if i < -1e-6)
    
    if Hess_idx == idx_order:
        # Near/after the saddle: Newton–Raphson step
        lam = 0.0
    else:
        eps = 1e-6
        a = k[idx_order - 1] + eps
        b = k[idx_order] - eps

        def f_lambda(lmbda):
            return np.sum(d**2 / (lmbda - k)**3)

        fa = f_lambda(a)
        fb = f_lambda(b)

        # If the bracket fails (e.g. numerical issues), fall back to Newton
        if np.isnan(fa) or np.isnan(fb):
            lam = 0.0
        else:
            # Bisection to solve f_lambda(λ) = 0
            for _ in range(80):
                m = 0.5 * (a + b)
                fm = f_lambda(m)
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            lam = 0.5 * (a + b)

        # Perturbation to prevent oscillation in the valley
        for i in range(g.size):
            if g[i] == 0:
                g[i] += 1e-3

    n = Q.size
    A = lam * np.eye(n) - H
    step = np.linalg.solve(A, g)

    # Limit step length for stability
    step_norm = np.linalg.norm(step)
    if step_norm > max_step:
        step *= (max_step / step_norm)

    return step, g, H, lam


running = False
Q_current = None
k_eff_current = None
beta_current = None
V0_current = V0_default
a_current = a_default
alpha_current = alpha_default
T_current = 150.0
n_beads_current = 128
idx_current = 1
gnorm_history = []
iter_count = 0
max_iter = 200
fd_step_default = 1e-4
max_step_default = 0.5
saddle_idx_default = 1

Tc_current = np.nan
x_barrier_current = np.nan
Vpp_barrier_current = np.nan

Q_bar_default = 0.0
Q_rad_default = 1.0
n_wind_default = 1

# x grid for plotting the external potential
x_plot_min = -5.0
x_plot_max = 5.0
x_plot = np.linspace(x_plot_min, x_plot_max, 400)


def V_external(Q):
    """Wrapper for the external potential using current slider parameters."""
    return V_Asym(Q, V_0=V0_current, a=a_current, alpha=alpha_current)

def V_profile(x):
    """Potential profile for plotting: V(x) with current parameters."""
    return V_Asym_local(x, V_0=V0_current, a=a_current, alpha=alpha_current)


def compute_turnover_temperature():
    """
    Numerically find the top of the external barrier and compute
    """
    global Tc_current, x_barrier_current, Vpp_barrier_current

    # Coarse search for maximum near x=0
    xs = np.linspace(-5.0, 5.0, 2001)
    Vvals = V_Asym_local(xs, V_0=V0_current, a=a_current, alpha=alpha_current)
    idx_max = np.argmax(Vvals)
    x_b = xs[idx_max]

    # Numerical second derivative at x_b
    h = 1e-3
    Vp = V_Asym_local(x_b + h, V_0=V0_current, a=a_current, alpha=alpha_current)
    Vm = V_Asym_local(x_b - h, V_0=V0_current, a=a_current, alpha=alpha_current)
    V0 = V_Asym_local(x_b, V_0=V0_current, a=a_current, alpha=alpha_current)
    Vpp = (Vp - 2.0 * V0 + Vm) / (h * h)

    x_barrier_current = x_b
    Vpp_barrier_current = Vpp

    if Vpp >= 0:
        Tc_current = np.nan
    else:
        omega_b = np.sqrt(-Vpp / m_mass)
        Tc_current = hbar * omega_b / (2.0 * np.pi * kB)

    print("\nBarrier search:")
    print(f"x_b ≈ {x_b:.4f}, V''(x_b) ≈ {Vpp:.4e}")
    if np.isfinite(Tc_current):
        print(f"T_c = ħ ω_b / (2π k_B) ≈ {Tc_current:.2f} K")
    else:
        print("T_c undefined (V''(x_b) ≥ 0)")

    return Tc_current


def make_initial_configuration():
    global Q_current, beta_current, k_eff_current, gnorm_history, iter_count
    global V0_current, a_current, alpha_current, idx_current, T_current, n_beads_current

    Q_bar = s_Q_center.val
    Q_rad = s_Q_radius.val
    n_wind = int(round(s_Q_nwind.val))
    T_current = s_temp.val
    n_beads_current = int(round(s_nbeads.val))
    V0_current = s_V0.val
    a_current = s_a.val
    alpha_current = s_alpha.val
    idx_current = int(round(s_idx.val))

    beta_current = 1.0 / (kB * T_current)
    k_eff_current = m_mass * (n_beads_current / (beta_current * hbar))**2

    k_indices = np.arange(n_beads_current)
    Q_current = Q_bar + Q_rad * np.sin(
        n_wind * 2 * np.pi * k_indices / n_beads_current
    )

    gnorm_history = []
    iter_count = 0

    compute_turnover_temperature()


# Figure 1: Q(k)
fig_Q = plt.figure("Q(k) vs bead index", figsize=(6, 4))
ax_Q = fig_Q.add_subplot(111)
ax_Q.set_title("Ring polymer configuration")
ax_Q.set_xlabel(r"Bead index $k$")
ax_Q.set_ylabel(r"$Q(k)$")

# Figure 2: beads on potential
fig_pot = plt.figure("Beads on potential", figsize=(6, 4))
ax_pot = fig_pot.add_subplot(111)
ax_pot.set_title("Beads on external potential")
ax_pot.set_xlabel(r"$Q_k$")
ax_pot.set_ylabel(r"$V(Q_k)$")

# Figure 3: gradient norm
fig_grad = plt.figure("Gradient norm", figsize=(6, 4))
ax_grad = fig_grad.add_subplot(111)
ax_grad.set_title("Norm of gradient vs iteration")
ax_grad.set_xlabel("Iteration")
ax_grad.set_ylabel(r"$|\nabla V_{\mathrm{polymer}}|$")

# Initial dummy configuration
k_indices = np.arange(n_beads_current)
Q_current = Q_bar_default + Q_rad_default * np.sin(
    n_wind_default * 2 * np.pi * k_indices / n_beads_current
)

# Plot objects
line_Q, = ax_Q.plot(k_indices, Q_current, marker='o', linestyle='-')
line_V, = ax_pot.plot(x_plot, V_profile(x_plot), lw=1.5)
scatter_Q = ax_pot.scatter(Q_current, V_profile(Q_current), color='C1')
line_grad, = ax_grad.plot([], [], marker='o', linestyle='-')

ax_Q.relim()
ax_Q.autoscale_view()
ax_pot.relim()
ax_pot.autoscale_view()
ax_grad.relim()
ax_grad.autoscale_view()


# Figure 4: sliders and buttons
fig_ctrl = plt.figure("Controls", figsize=(8, 6))
axcolor = 'lightgoldenrodyellow'

# Sliders
ax_Q_center = fig_ctrl.add_axes([0.10, 0.72, 0.35, 0.03], facecolor=axcolor)
ax_Q_radius = fig_ctrl.add_axes([0.10, 0.66, 0.35, 0.03], facecolor=axcolor)
ax_Q_nwind = fig_ctrl.add_axes([0.10, 0.60, 0.35, 0.03], facecolor=axcolor)
ax_idx = fig_ctrl.add_axes([0.10, 0.54, 0.35, 0.03], facecolor=axcolor)

s_Q_center = Slider(ax_Q_center, r"$Q$ center", -2.0, 2.0, valinit=Q_bar_default)
s_Q_radius = Slider(ax_Q_radius, r"$Q$ radius", 0.1, 3.0, valinit=Q_rad_default)
s_Q_nwind = Slider(ax_Q_nwind, r"$n_{\mathrm{wind}}$", 1, 9, valinit=n_wind_default, valstep=1)
s_idx = Slider(ax_idx, "index", 1, 9, valinit=saddle_idx_default, valstep=1)

ax_V0 = fig_ctrl.add_axes([0.60, 0.72, 0.30, 0.03], facecolor=axcolor)
ax_a = fig_ctrl.add_axes([0.60, 0.66, 0.30, 0.03], facecolor=axcolor)
ax_alpha = fig_ctrl.add_axes([0.60, 0.60, 0.30, 0.03], facecolor=axcolor)

s_V0 = Slider(ax_V0, r"$V_0$ ($E_h$)", 0.001, 0.05, valinit=V0_default)
s_a = Slider(ax_a, r"$a$", 0.2, 2.0, valinit=a_default)
s_alpha = Slider(ax_alpha, r"$\alpha$", 0.5, 4.0, valinit=alpha_default)

ax_temp = fig_ctrl.add_axes([0.10, 0.46, 0.35, 0.03], facecolor=axcolor)
ax_nbeads = fig_ctrl.add_axes([0.60, 0.46, 0.30, 0.03], facecolor=axcolor)

s_temp = Slider(ax_temp, r"$T$ (K)", 50.0, 500.0, valinit=T_current)
s_nbeads = Slider(ax_nbeads, r"$n$ beads", 16, 256,
                  valinit=n_beads_current, valstep=8)

# Buttons
ax_start = fig_ctrl.add_axes([0.15, 0.25, 0.20, 0.06])
ax_reset = fig_ctrl.add_axes([0.55, 0.25, 0.30, 0.06])

button_start = Button(ax_start, "Start", color='lightgreen', hovercolor='0.9')
button_reset = Button(ax_reset, "Reset && Apply",
                      color='lightcoral', hovercolor='0.9')


def reset_parameters(event):
    """
    Reset button callback:
    - Read slider values
    - Rebuild initial polymer configuration
    - Recompute T_c
    - Clear the three plots
    """
    global running, gnorm_history, iter_count, Q_current

    running = False

    make_initial_configuration()

    # Update the Q(k) plot
    k_indices = np.arange(n_beads_current)
    line_Q.set_data(k_indices, Q_current)
    ax_Q.set_xlim(0, max(1, n_beads_current - 1))
    ax_Q.relim()
    ax_Q.autoscale_view()

    # Update the potential curve
    line_V.set_data(x_plot, V_profile(x_plot))

    # Update bead scatter on potential
    Q_vals = Q_current
    V_vals = V_profile(Q_vals)
    scatter_Q.set_offsets(np.column_stack((Q_vals, V_vals)))
    ax_pot.relim()
    ax_pot.autoscale_view()

    # Update T_c
    if np.isfinite(Tc_current):
        title = f"Beads on external potential V(Q)\nT_c ≈ {Tc_current:.2f} K"
    else:
        title = "Beads on external potential V(Q)\nT_c undefined"
    ax_pot.set_title(title)

    # Clear gradient history
    gnorm_history = []
    iter_count = 0
    line_grad.set_data([], [])
    ax_grad.relim()
    ax_grad.autoscale_view()

    fig_Q.canvas.draw_idle()
    fig_pot.canvas.draw_idle()
    fig_grad.canvas.draw_idle()
    fig_ctrl.canvas.draw_idle()


def start_run(event):
    """
    Start button callback: start / resume the saddle search.
    (The script does NOT start running until this is pressed.)
    """
    global running
    if Q_current is None:
        make_initial_configuration()
        reset_parameters(None)
    running = True


button_start.on_clicked(start_run)
button_reset.on_clicked(reset_parameters)


def update_iteration():
    """
    Timer callback: perform one CM step and update the plots.
    Only runs while 'running' is True.
    """
    global Q_current, iter_count, gnorm_history, running

    if not running:
        return
    if Q_current is None:
        return

    def V_ext(Q):
        return V_external(Q)

    step, g, H, lam = Polymer_step(
        Q_current, k_eff_current, V_ext, idx_current,
        fd_step=fd_step_default,
        max_step=max_step_default
    )
    Q_current = Q_current + step
    gnorm = np.linalg.norm(g)
    gnorm_history.append(gnorm)
    iter_count += 1

    Hess_evals = np.linalg.eigvalsh(H)

    print(f"iteration {iter_count}: |grad V(Q)| = {gnorm:.3e}")
    Saddle_idx = sum(1 for i in Hess_evals if i < -1e-6)
    print("Hessian eigenvalues:",
        np.array2string(Hess_evals,
                        formatter={'float_kind': lambda x: f"{x:.3e}"},
                        threshold=10,
                        edgeitems=Saddle_idx + 3))

    k_indices = np.arange(n_beads_current)
    line_Q.set_data(k_indices, Q_current)
    ax_Q.set_xlim(0, max(1, n_beads_current - 1))
    ax_Q.relim()
    ax_Q.autoscale_view()

    Q_vals = Q_current
    V_vals = V_profile(Q_vals)
    scatter_Q.set_offsets(np.column_stack((Q_vals, V_vals)))
    ax_pot.relim()
    ax_pot.autoscale_view()

    x_iter = np.arange(len(gnorm_history))
    line_grad.set_data(x_iter, np.array(gnorm_history))
    ax_grad.set_xlim(0, max(10, len(gnorm_history)))
    if len(gnorm_history) > 0:
        ax_grad.set_ylim(0, max(gnorm_history) * 1.1)

    fig_Q.canvas.draw_idle()
    fig_pot.canvas.draw_idle()
    fig_grad.canvas.draw_idle()

    # Stop if converged or hit max iterations
    if gnorm < 1e-10:
        print("\n=== Saddle point found after", iter_count, "iterations ===")
        print(f"|grad V(Q*)| = {gnorm:.3e}")
        print("Hessian eigenvalues:",
              np.array2string(Hess_evals,
                              formatter={'float_kind': lambda x: f"{x:.3e}"},
                              threshold=10,
                              edgeitems=Saddle_idx + 3))
        print("Saddle index:", Saddle_idx)
        running = False
    elif iter_count >= max_iter:
        print("\n=== Saddle point not found (max_iter reached) ===")
        running = False


# Timer: calls update_iteration repeatedly; attach to one of the figures
timer = fig_Q.canvas.new_timer(interval=70)  # ms
timer.add_callback(update_iteration)
timer.start()

make_initial_configuration()
reset_parameters(None)

plt.show()
