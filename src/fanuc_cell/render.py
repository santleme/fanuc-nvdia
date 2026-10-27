"""Matplotlib renderer for a headless kinematic proof-of-concept clip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .task import PickPlaceEpisode


def render_episode(episode: PickPlaceEpisode, output: str | Path, *, fps: int = 24) -> Path:
    """Render the scripted cycle to MP4 or GIF; this is visualization, not physics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(10, 7), layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    axis.set(xlim=(-0.1, 1.6), ylim=(-0.9, 0.9), zlim=(0, 1.5))
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.view_init(elev=23, azim=-54)

    config = episode.config
    xx, yy = np.meshgrid(
        np.linspace(config.belt_start_x - 0.08, config.belt_end_x + 0.05, 2),
        np.linspace(-0.42, 0.42, 2),
    )
    zz = np.full_like(xx, config.belt_surface_z)
    axis.plot_surface(xx, yy, zz, color="#475569", alpha=0.72, shade=True)
    axis.plot(
        [config.belt_start_x, config.belt_end_x],
        [config.belt_y, config.belt_y],
        [config.belt_surface_z + 0.005] * 2,
        color="#e2e8f0",
        linewidth=3,
    )

    place = np.asarray(config.place_xyz)
    axis.scatter(*place, marker="x", s=80, color="#22c55e", label="Deposit")
    robot_line, = axis.plot([], [], [], color="#facc15", linewidth=9, solid_capstyle="round")
    joint_points = axis.scatter([], [], [], s=40, color="#f8fafc", depthshade=False)
    part = axis.scatter([], [], [], s=250, marker="s", color="#38bdf8", edgecolor="#0f172a")
    gripper_lines, = axis.plot([], [], [], color="#ef4444", linewidth=4)
    state_label = axis.text2D(0.02, 0.96, "", transform=axis.transAxes, fontsize=11)
    axis.legend(loc="upper right")

    def update(frame: int):
        time = frame / fps
        joints, part_position, closed, phase, success = episode.sample(time)
        tool, points = episode.arm.forward(joints)
        coords = np.asarray(points)
        robot_line.set_data(coords[:, 0], coords[:, 1])
        robot_line.set_3d_properties(coords[:, 2])
        joint_points._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
        part._offsets3d = ([part_position[0]], [part_position[1]], [part_position[2]])

        finger_gap = 0.055 if not closed else 0.03
        fingers = np.array(
            [
                [tool[0], tool[0]],
                [tool[1] - finger_gap, tool[1] + finger_gap],
                [tool[2] - 0.09, tool[2] - 0.09],
            ]
        )
        gripper_lines.set_data(fingers[0], fingers[1])
        gripper_lines.set_3d_properties(fingers[2])
        state_label.set_text(
            f"M-10iD/12 proxy | {phase.upper()} | t={time:.2f}s | "
            f"gripper={'closed' if closed else 'open'} | {'SUCCESS' if success else 'RUNNING'}"
        )
        return robot_line, joint_points, part, gripper_lines, state_label

    frame_count = int(np.ceil(episode.duration * fps)) + 1
    movie = FuncAnimation(figure, update, frames=frame_count, interval=1000 / fps, blit=False)
    if destination.suffix.lower() == ".gif":
        movie.save(destination, writer=PillowWriter(fps=fps))
    elif destination.suffix.lower() == ".mp4":
        try:
            import imageio_ffmpeg

            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        try:
            movie.save(destination, writer=FFMpegWriter(fps=fps, codec="libx264", bitrate=1800))
        except Exception as ffmpeg_error:
            # Some managed notebook/workstation images ship a broken system
            # ffmpeg. OpenCV's bundled encoder is a usable MP4 fallback.
            destination.unlink(missing_ok=True)
            try:
                import cv2

                encoder = cv2.VideoWriter(
                    str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, figure.canvas.get_width_height()
                )
                if not encoder.isOpened():
                    raise RuntimeError("OpenCV could not initialize an MP4 writer")
                try:
                    for frame in range(frame_count):
                        update(frame)
                        figure.canvas.draw()
                        rgba = np.asarray(figure.canvas.buffer_rgba())
                        encoder.write(cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))
                finally:
                    encoder.release()
            except ImportError:
                raise RuntimeError(
                    "MP4 encoding failed. Install the project video extra or opencv-python-headless."
                ) from ffmpeg_error
            except Exception as cv_error:
                destination.unlink(missing_ok=True)
                raise RuntimeError(f"MP4 encoding failed with ffmpeg and OpenCV: {cv_error}") from ffmpeg_error
    else:
        raise ValueError("Video output extension must be .mp4 or .gif")
    plt.close(figure)
    return destination
