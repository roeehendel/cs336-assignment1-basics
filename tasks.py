import os
from datetime import datetime

from invoke import Context, task


@task
def profile(c: Context, script_path: str) -> None:
    """Profile a script using viztracer."""

    script_filename = os.path.basename(script_path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = f"profiling/vizviewer-{script_filename}-{timestamp}.json"

    os.makedirs("profiling", exist_ok=True)

    c.run(
        f"viztracer "
        f"--exclude_files '/opt/homebrew/' '/usr/' '/Library/' "
        f"--min_duration 50 "
        f"--max_stack_depth 20 "
        f"--tracer_entries 20000000 "
        f"-o {output_path} "
        f"{script_path}"
    )


@task
def test(c: Context, profile: bool = False) -> None:
    """Run the tests."""
    command = "uv run pytest ."

    if profile:
        command += " --durations=10"

    c.run(command, pty=True)
