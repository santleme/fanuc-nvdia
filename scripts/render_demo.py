from __future__ import annotations

import argparse
import json
from pathlib import Path

from fanuc_cell import make_demo_episode
from fanuc_cell.render import render_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a FANUC-style conveyor pick-and-place cycle")
    parser.add_argument("--output", default="outputs/fanuc-cell-demo.mp4")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    episode = make_demo_episode()
    path = render_episode(episode, args.output, fps=args.fps)
    joints, part, closed, phase, success = episode.sample(episode.duration)
    grasp_time = next(point.time for point in episode.waypoints if point.label == "grasp")
    _, intercept_part, _, _, _ = episode.sample(grasp_time)
    report_path = Path(args.output).with_suffix(".json")
    report = {
        "model": episode.arm.model.name,
        "mode": "kinematic_reference_no_contact_dynamics",
        "duration_seconds": episode.duration,
        "waypoint_count": len(episode.waypoints),
        "grasp_intercept_error_m": float(abs(intercept_part[0] - episode.config.pick_x)),
        "final_part_position_m": [float(value) for value in part],
        "target_place_position_m": list(episode.config.place_xyz),
        "final_joint_positions_rad": [float(value) for value in joints],
        "gripper_closed_at_end": bool(closed),
        "final_phase": phase,
        "success": bool(success),
        "video": str(path.resolve()),
    }
    assert success and not closed, "Reference episode did not finish with the part released"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"VIDEO_PATH={path.resolve()}")
    print(f"REPORT_PATH={report_path.resolve()}")


if __name__ == "__main__":
    main()
