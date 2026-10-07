"""
Open a model in the interactive MuJoCo viewer.

    python view.py                                    # default project
    python view.py --project double_pendulum
    python view.py --xml path/to/any_model.xml
"""
import argparse

import mujoco
import mujoco.viewer

from projects import DEFAULT_PROJECT, PROJECTS, get_project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT, choices=sorted(PROJECTS))
    parser.add_argument("--xml", default=None, help="view this file instead of a project's model")
    args = parser.parse_args()

    path = args.xml or get_project(args.project).model_path
    print(f"viewing {path}")
    mujoco.viewer.launch(mujoco.MjModel.from_xml_path(str(path)))


if __name__ == "__main__":
    main()
