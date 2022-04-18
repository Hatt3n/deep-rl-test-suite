"""
This file was created by John Wikman, who graciously gave his permission
to include it as a part of this repository. Minor modifications have been
made by @dansah.
"""
import math
import os
from matplotlib import projections

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def _plot_3D_axes(ax, phi, theta, l_arm, l_pendulum):
    """Internal function used to plot a furuta pendulum on a 3D Axes."""

    ax.clear()

    # (correction, theta is inverted from the model)
    theta = -theta

    # Assume that we are looking straight on to point on the arm while at a
    # distance from the DC motor. Then theta is the angle from the positive
    # y-axis towards the positive x-axis:
    #         . y
    #        /|\
    #         |
    #         |θ/
    #         |/
    #  ---------------> x
    #         |
    #         |
    #         |
    #
    # Starting at the position where phi = 0, the pendulum base is located at
    # x=l_arm, y,z=0. Then x turns to y and y turn to z in the plot above.
    initial_pendulum_tip_coord = [
        l_arm,
        l_pendulum * np.cos((np.pi / 2) - theta),
        l_pendulum * np.sin((np.pi / 2) - theta)
    ]

    # Arm is in a level plane looking from above. Assume phi is angle from
    # x-axis in direction towards the positive y-axis (i.e. unit circle)
    # Now apply transformation to rotate the frame in accordance with phi. Our
    # new basis vectors are:
    phi_matrix = np.array([
        [np.cos(phi),  np.sin(phi), 0.0], # x-vector
        [-np.sin(phi), np.cos(phi), 0.0], # y-vector
        [0.0,          0.0,         1.0]  # z-vector (unaffected)
    ])

    initial_coords = np.array([
        [0.0,   0.0, 0.0], # origin
        [l_arm, 0.0, 0.0], # arm tip when phi=0
        initial_pendulum_tip_coord
    ])
    
    coords = np.transpose(np.matmul(
        phi_matrix,
        np.transpose(initial_coords)
    ))

    def split(coord_list):
        xs = [c[0] for c in coord_list]
        ys = [c[1] for c in coord_list]
        zs = [c[2] for c in coord_list]
        return xs, ys, zs

    ax.plot3D(*split(coords[0:2]), label="Arm", linewidth=5, color="blue")
    ax.plot3D(*split(coords[1:3]), label="Pendulum", linewidth=3, color="orange")

    ax.set_xlim([-abs(l_arm) - 0.2, abs(l_arm) + 0.2])
    ax.set_ylim([-abs(l_arm) - 0.2, abs(l_arm) + 0.2])
    ax.set_zlim([-abs(l_pendulum) - 0.2, abs(l_pendulum) + 0.2])
    ax.legend()


def _plot_polar_indicator(ax, angle, title=None, color="red"):
    """Plot the arm angle in a flat plane"""
    ax.clear()
    ax.plot([angle, angle], [0.0, 1.0], color=color, linewidth=2)
    ax.set_yticks([])
    ax.set_theta_offset(np.pi / 2)

    if title is not None:
        ax.set_title(title)


def _plot_timeseries(ax, ts, ys, title=None, ylim=None, color="red"):
    """Plots a regular rectilinear plot"""
    ax.clear()
    ax.plot(ts, ys, color=color)
    ax.set_xlabel("t [s]")
    ax.grid()

    if ylim is not None:
        ax.set_ylim(bottom=min(ylim), top=max(ylim))
    if title is not None:
        ax.set_title(title)


def plot_static(phi, theta, l_arm=2.0, l_pendulum=1.0):
    """Plots a static furuta pendulum for a single moment in time."""
    fig = plt.figure()
    fig.clear()
    ax = fig.add_subplot(projection="3d")

    _plot_3D_axes(ax, phi, theta, l_arm, l_pendulum)

    plt.show()


def plot_animated(phis, thetas,
                  us=None, dphis=None, dthetas=None, ts=None,
                  l_arm=2.0, l_pendulum=1.0,
                  frame_rate=60, save_as=None, show_extra_details=False):
    """Display an animated plot of a furuta pendulum."""

    fig = plt.figure(figsize=(16, 9))
    fig.clear()
    if show_extra_details: # Argument added by @dansah. Hides everything except the arm and pendulum.
        ax3d = fig.add_subplot(2, 5, (1, 2), projection="3d")
        ax_polar_phi = fig.add_subplot(2, 5, 3, projection="polar")
        ax_polar_theta = fig.add_subplot(2, 5, 5, projection="polar")
        ax_u = fig.add_subplot(2, 5, 6, projection="rectilinear")
        ax_phi = fig.add_subplot(2, 5, 7, projection="rectilinear")
        ax_theta = fig.add_subplot(2, 5, 8, projection="rectilinear")
        ax_dphidt = fig.add_subplot(2, 5, 9, projection="rectilinear")
        ax_dthetadt = fig.add_subplot(2, 5, 10, projection="rectilinear")
    else:
        ax3d = fig.add_subplot(1, 1, (1, 1), projection="3d")

    if len(phis) != len(thetas):
        raise ValueError(f"Unequal input lengths len(phis) != len(thetas) ({len(phis)} != {len(thetas)})")

    N_FRAMES = len(phis)

    if ts is None:
        ts = [float(i / frame_rate) for i in range(N_FRAMES)] # Modified to divide by frame_rate and start at 0. @dansah

    if us is None:
        us = [0.0] * N_FRAMES
    if dphis is None:
        dphis = [0.0] * N_FRAMES
    if dthetas is None:
        dthetas = [0.0] * N_FRAMES

    def iterate(frame_idx):
        phi = phis[frame_idx]
        theta = thetas[frame_idx]
        t = ts[frame_idx]
        _plot_3D_axes(ax3d, phi, theta, l_arm, l_pendulum)
        if show_extra_details:
            _plot_polar_indicator(ax_polar_phi, phi, title="Arm Angle", color="blue")
            _plot_polar_indicator(ax_polar_theta, theta, title="Pendulum Angle", color="orange")
            # Keep last 4 seconds of parameters
            i_lower = max(0, frame_idx - int(frame_rate)*4)
            i_upper = frame_idx + 1
            _plot_timeseries(ax_u,
                            ts[i_lower:i_upper],
                            us[i_lower:i_upper],
                            ylim=(min(us), max(us)),
                            title="Control input",
                            color="black")
            _plot_timeseries(ax_phi,
                            ts[i_lower:i_upper],
                            phis[i_lower:i_upper],
                            ylim=(min(phis), max(phis)),
                            title="Phi",
                            color="blue")
            _plot_timeseries(ax_theta,
                            ts[i_lower:i_upper],
                            thetas[i_lower:i_upper],
                            ylim=(min(thetas), max(thetas)),
                            title="Theta",
                            color="orange")
            _plot_timeseries(ax_dphidt,
                            ts[i_lower:i_upper],
                            dphis[i_lower:i_upper],
                            ylim=(min(dphis), max(dphis)),
                            title="dPhi/dt",
                            color="red")
            _plot_timeseries(ax_dthetadt,
                            ts[i_lower:i_upper],
                            dthetas[i_lower:i_upper],
                            ylim=(min(dthetas), max(dthetas)),
                            title="dTheta/dt",
                            color="darkgreen")
        ax3d.set_title(f"Furuta Pendulum visualization (t = {t:.2f} seconds)")

    ani = animation.FuncAnimation(
        fig,
        iterate,
        fargs=(),
        frames=N_FRAMES,
        interval=(1000/frame_rate),
        repeat=False
    )
    # Save animation as video?
    if save_as is not None:
        mp4_writer = animation.FFMpegWriter(fps=frame_rate, codec="h264")

        contents = os.listdir(".")
        filename = f"{save_as}.mp4"
        f_suffix = 1
        while filename in contents:
            filename = f"{save_as}-{f_suffix}.mp4"
            f_suffix += 1

        print(f"Saving as {filename}... (this may take a couple of minutes)")
        ani.save(filename,
                 writer=mp4_writer)
        print("Movie clip saved.")
    else:
        plt.show()

