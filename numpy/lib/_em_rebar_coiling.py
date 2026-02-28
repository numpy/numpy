"""
Multiphysics Framework for Electromagnetic Rebar Coiling Animation.

This module provides numpy-based tools to simulate and generate animation
frames for electromagnetic rebar coiling processes, modelling the coupled
electromagnetic, mechanical, and thermal physics.

Physical model assumptions and limitations
------------------------------------------
* **Magnetic field** – evaluated with a single circular current-loop
  Biot-Savart approximation.  The on-axis component uses the exact
  closed-form expression; the radial component is an approximate
  dipole correction.  Near the coil wire itself the field diverges
  and the approximation breaks down.
* **Time integration** – uses an explicit first-order Euler scheme,
  which can be unstable for stiff problems.  Choose ``dt`` small enough
  that the highest-frequency mechanical mode is well-resolved
  (Courant-type condition).  For engineering accuracy, replace with a
  velocity-Verlet or RK4 integrator.
* **Elastic restoring force** – a simple linear spring penalty drives the
  rebar toward a target helical shape; it does not represent full 3-D
  continuum mechanics.
* **Joule heating** – assumes a uniform, temperature-independent
  resistivity.  Thermal conduction between elements is not modelled.
* The simulation is intended for qualitative visualisation of the coiling
  process, not quantitative engineering design.
"""

import numpy as np

__all__ = [
    "compute_magnetic_field",
    "compute_lorentz_force",
    "compute_joule_heating",
    "simulate_rebar_coiling",
    "generate_coil_animation_frames",
]

# Spring damping scale factor applied to the elastic restoring force.
# This dimensionless value prevents numerical blow-up caused by the very
# large stiffness (k = EA/L ~ O(10^9) N/m) relative to the small time
# steps used in the explicit Euler integrator.
_SPRING_DAMPING_SCALE = 1e-6


def compute_magnetic_field(current, positions, coil_center, coil_radius):
    """
    Compute the magnetic field produced by a circular current loop using
    the Biot-Savart law approximation.

    Parameters
    ----------
    current : float
        Electric current in Amperes.
    positions : ndarray, shape (N, 3)
        Spatial positions at which to evaluate the field.
    coil_center : array_like, shape (3,)
        Center of the current loop [x, y, z].
    coil_radius : float
        Radius of the current loop in metres.

    Returns
    -------
    B_field : ndarray, shape (N, 3)
        Magnetic flux density vector [Bx, By, Bz] in Tesla at each position.
    """
    mu_0 = 4.0 * np.pi * 1e-7  # permeability of free space (H/m)
    positions = np.asarray(positions, dtype=float)
    coil_center = np.asarray(coil_center, dtype=float)

    r_vec = positions - coil_center  # (N, 3)
    r_mag = np.linalg.norm(r_vec, axis=-1, keepdims=True)  # (N, 1)

    # Avoid singularity at the coil centre
    r_mag = np.where(r_mag < 1e-12, 1e-12, r_mag)

    # On-axis approximation: B_z component of a magnetic dipole loop
    # B ≈ (mu_0 * I * R^2) / (2 * (R^2 + z^2)^(3/2))  (axial component)
    z_comp = r_vec[:, 2:3]  # axial distance
    rho_sq = r_vec[:, 0:1] ** 2 + r_vec[:, 1:2] ** 2  # radial distance^2
    denom = (coil_radius ** 2 + z_comp ** 2) ** 1.5
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

    B_z = (mu_0 * current * coil_radius ** 2) / (2.0 * denom)

    # Radial component (pointing away from axis towards wire position)
    rho = np.sqrt(rho_sq)
    safe_rho = np.where(rho > 1e-12, rho, 1.0)  # avoid division by zero
    B_rho = np.where(
        rho > 1e-12,
        -(mu_0 * current * coil_radius ** 2 * z_comp)
        / (4.0 * denom * safe_rho),
        0.0,
    )

    # Convert cylindrical (rho, phi, z) -> Cartesian
    phi = np.arctan2(r_vec[:, 1:2], r_vec[:, 0:1])
    B_x = B_rho * np.cos(phi)
    B_y = B_rho * np.sin(phi)

    return np.hstack([B_x, B_y, B_z])


def compute_lorentz_force(current_density, B_field, volume):
    """
    Compute the Lorentz body force on a current-carrying volume element.

    Parameters
    ----------
    current_density : ndarray, shape (N, 3)
        Current density vector J [A/m^2] at each element.
    B_field : ndarray, shape (N, 3)
        Magnetic flux density B [T] at each element.
    volume : ndarray, shape (N,) or float
        Volume of each element in m^3.

    Returns
    -------
    force : ndarray, shape (N, 3)
        Lorentz force [N] on each element: F = J x B * volume.
    """
    current_density = np.asarray(current_density, dtype=float)
    B_field = np.asarray(B_field, dtype=float)
    volume = np.asarray(volume, dtype=float)

    # Cross product J × B for each element
    j_cross_b = np.cross(current_density, B_field)  # (N, 3)
    return j_cross_b * volume[:, np.newaxis]


def compute_joule_heating(current_density, resistivity, volume, dt):
    """
    Compute Joule heating in a resistive conductor element.

    Parameters
    ----------
    current_density : ndarray, shape (N, 3)
        Current density vector J [A/m^2].
    resistivity : float or ndarray, shape (N,)
        Electrical resistivity [Ohm·m].
    volume : ndarray, shape (N,) or float
        Volume of each element [m^3].
    dt : float
        Time step [s].

    Returns
    -------
    heat : ndarray, shape (N,)
        Heat energy generated per element [J] in time dt.
    """
    current_density = np.asarray(current_density, dtype=float)
    J_sq = np.sum(current_density ** 2, axis=-1)  # |J|^2 (N,)
    return resistivity * J_sq * volume * dt


def simulate_rebar_coiling(
    rebar_length,
    rebar_radius,
    n_elements,
    coil_radius,
    current,
    resistivity,
    density,
    youngs_modulus,
    n_steps,
    dt,
    specific_heat=490.0,
):
    """
    Simulate rebar coiling driven by electromagnetic Lorentz forces.

    The rebar is discretised into ``n_elements`` cylindrical segments.
    At each time step the magnetic field produced by the forming coil is
    evaluated, the Lorentz force on every segment is computed and the
    resulting displacement is integrated using a simple explicit Euler
    scheme (first-order; choose ``dt`` small enough for stability).

    Parameters
    ----------
    rebar_length : float
        Total length of the rebar [m].
    rebar_radius : float
        Cross-sectional radius of the rebar [m].
    n_elements : int
        Number of discrete elements along the rebar.
    coil_radius : float
        Target coil radius [m].
    current : float
        Applied current [A].
    resistivity : float
        Electrical resistivity of the rebar material [Ohm·m].
    density : float
        Mass density of the rebar material [kg/m^3].
    youngs_modulus : float
        Young's modulus of the rebar material [Pa].
    n_steps : int
        Number of time integration steps.
    dt : float
        Time step size [s].  Must be small enough for the explicit Euler
        integrator to remain stable.
    specific_heat : float, optional
        Specific heat capacity of the rebar material [J/(kg·K)].
        Default is 490 J/(kg·K), a typical value for structural steel.

    Returns
    -------
    positions : ndarray, shape (n_steps + 1, n_elements, 3)
        3-D positions of each element centroid at each time step.
    temperatures : ndarray, shape (n_steps + 1, n_elements)
        Temperature of each element at each time step [K] (relative rise
        above ambient; ambient = 0).
    """
    rebar_length = float(rebar_length)
    rebar_radius = float(rebar_radius)
    n_elements = int(n_elements)
    coil_radius = float(coil_radius)

    # Element properties
    elem_length = rebar_length / n_elements
    cross_area = np.pi * rebar_radius ** 2
    elem_volume = cross_area * elem_length
    elem_mass = density * elem_volume

    # Initial straight-rod configuration along the x-axis
    x0 = np.linspace(elem_length / 2, rebar_length - elem_length / 2, n_elements)
    positions_all = np.zeros((n_steps + 1, n_elements, 3))
    positions_all[0, :, 0] = x0

    temperatures_all = np.zeros((n_steps + 1, n_elements))

    # Velocity array
    velocities = np.zeros((n_elements, 3))

    # Coil centre (fixed at origin)
    coil_center = np.array([rebar_length / 2, 0.0, 0.0])

    for step in range(n_steps):
        pos = positions_all[step]  # (N, 3)
        temp = temperatures_all[step].copy()

        # Current density (uniform along x-axis of each segment initially)
        J = np.zeros((n_elements, 3))
        J[:, 0] = current / cross_area

        # Magnetic field at each element centroid
        B = compute_magnetic_field(current, pos, coil_center, coil_radius)

        # Lorentz forces on each element
        F = compute_lorentz_force(J, B, np.full(n_elements, elem_volume))

        # Simple elastic restoring force (resist over-deformation)
        # proportional to displacement from the target helical shape
        target = _target_helix(n_elements, coil_radius, rebar_length)
        displacement = pos - target
        k_spring = youngs_modulus * cross_area / elem_length
        F_restore = -k_spring * displacement * _SPRING_DAMPING_SCALE

        F_total = F + F_restore

        # Euler integration: a = F / m
        accel = F_total / elem_mass
        velocities = velocities + accel * dt
        new_pos = pos + velocities * dt

        # Joule heating
        dQ = compute_joule_heating(
            J, resistivity, np.full(n_elements, elem_volume), dt
        )
        dT = dQ / (elem_mass * specific_heat)
        temp += dT

        positions_all[step + 1] = new_pos
        temperatures_all[step + 1] = temp

    return positions_all, temperatures_all


def _target_helix(n_elements, coil_radius, rebar_length):
    """Return target helical coil positions for ``n_elements`` element centroids.

    The helix makes exactly one full turn (0 to 2π) about the z-axis.
    The pitch (axial advance per radian) equals ``rebar_length / (2π)``,
    ensuring the total arc length approximates ``rebar_length``.

    Parameters
    ----------
    n_elements : int
        Number of element centroids.
    coil_radius : float
        Radius of the helix [m].
    rebar_length : float
        Total length of the rebar [m], used to determine pitch.

    Returns
    -------
    positions : ndarray, shape (n_elements, 3)
        Cartesian coordinates [x, y, z] of each target centroid.
    """
    t = np.linspace(0, 2 * np.pi, n_elements)
    pitch = rebar_length / (2 * np.pi)
    x = coil_radius * np.cos(t)
    y = coil_radius * np.sin(t)
    z = pitch * t
    return np.stack([x, y, z], axis=-1)


def generate_coil_animation_frames(
    positions,
    temperatures,
    frame_stride=1,
):
    """
    Extract animation frames from a coiling simulation result.

    Parameters
    ----------
    positions : ndarray, shape (n_steps + 1, n_elements, 3)
        Element positions from :func:`simulate_rebar_coiling`.
    temperatures : ndarray, shape (n_steps + 1, n_elements)
        Element temperatures from :func:`simulate_rebar_coiling`.
    frame_stride : int, optional
        Stride for down-sampling frames (default: 1, i.e., every step).

    Returns
    -------
    frames : list of dict
        Each element is a frame dictionary with keys:

        ``"positions"``
            ndarray, shape (n_elements, 3) – element centroids.
        ``"temperatures"``
            ndarray, shape (n_elements,) – element temperatures.
        ``"step"``
            int – simulation step index for this frame.
    """
    frame_stride = int(frame_stride)
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    n_steps_plus1 = positions.shape[0]
    frames = []
    for idx in range(0, n_steps_plus1, frame_stride):
        frames.append(
            {
                "positions": positions[idx].copy(),
                "temperatures": temperatures[idx].copy(),
                "step": idx,
            }
        )
    return frames
