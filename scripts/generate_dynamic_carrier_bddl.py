#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu
from libero.libero.utils.task_generation_utils import (
    generate_bddl_from_task_info,
    register_task_info,
)

TARGET_XY = [0.18, 0.22]


@register_mu(scene_type="ttt_dynamic")
class DynamicCarrierFlat(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {"kitchen_table": 1}
        object_num_info = {
            "plate": 1,
            "cream_cheese": 1,
        }
        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.12, 0.0],
                region_name="carrier_init_region",
                target_name=self.workspace_name,
                region_half_len=0.015,
                yaw_rotation=(0.0, 0.0),
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=TARGET_XY,
                region_name="target_region",
                target_name=self.workspace_name,
                region_half_len=0.045,
                yaw_rotation=(0.0, 0.0),
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        return [
            ("On", "plate_1", "kitchen_table_carrier_init_region"),
            ("On", "cream_cheese_1", "plate_1"),
        ]


@register_mu(scene_type="ttt_dynamic")
class DynamicCarrierOpenBox(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {"kitchen_table": 1}
        object_num_info = {
            "wooden_tray": 1,
            "cream_cheese": 1,
        }
        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.12, 0.0],
                region_name="carrier_init_region",
                target_name=self.workspace_name,
                region_half_len=0.015,
                yaw_rotation=(0.0, 0.0),
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=TARGET_XY,
                region_name="target_region",
                target_name=self.workspace_name,
                region_half_len=0.045,
                yaw_rotation=(0.0, 0.0),
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        return [
            ("On", "wooden_tray_1", "kitchen_table_carrier_init_region"),
            ("In", "cream_cheese_1", "wooden_tray_1_contain_region"),
        ]


def register_tasks() -> None:
    register_task_info(
        language="pick up the cream cheese box from the moving flat platform and place it on the target region",
        scene_name="dynamic_carrier_flat",
        objects_of_interest=["plate_1", "cream_cheese_1"],
        goal_states=[("On", "cream_cheese_1", "kitchen_table_target_region")],
    )
    register_task_info(
        language="pick up the cream cheese box from the moving open box and place it on the target region",
        scene_name="dynamic_carrier_open_box",
        objects_of_interest=["wooden_tray_1", "cream_cheese_1"],
        goal_states=[("On", "cream_cheese_1", "kitchen_table_target_region")],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate starter BDDL files for dynamic carrier tasks.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "generated_bddl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    register_tasks()
    bddl_file_names, failures = generate_bddl_from_task_info(folder=str(args.output_dir))
    print("Generated BDDL files:")
    for file_name in bddl_file_names:
        print(file_name)
    if failures:
        print("Failures:")
        for scene_name, language in failures:
            print(f"- {scene_name}: {language}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
