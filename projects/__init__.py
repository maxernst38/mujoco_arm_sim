from projects import double_pendulum

PROJECTS = {p.name: p for p in (double_pendulum.PROJECT,)}
DEFAULT_PROJECT = "double_pendulum"


def get_project(name):
    if name not in PROJECTS:
        raise SystemExit(f"unknown project '{name}', available: {', '.join(sorted(PROJECTS))}")
    return PROJECTS[name]
