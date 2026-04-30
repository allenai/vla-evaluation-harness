"""Docker subprocess helpers."""

from __future__ import annotations

import subprocess
import sys

from vla_eval.cli._console import stderr_console as _stderr_console


def check_docker_daemon(docker: str) -> None:
    """Exit 1 with a clear message if the docker daemon is unreachable."""
    if subprocess.run([docker, "info"], capture_output=True).returncode != 0:
        _stderr_console().print(
            "[red]ERROR: Docker daemon is not running.[/red]\n  Start it with: sudo systemctl start docker",
        )
        sys.exit(1)


def image_exists_locally(docker: str, image: str) -> bool:
    """Return True if a docker image is present in the local store."""
    return subprocess.run([docker, "image", "inspect", image], capture_output=True).returncode == 0


def ensure_image_local(docker: str, image: str, auto_yes: bool) -> None:
    """Make sure ``image`` is available locally, prompting for ``docker pull`` when missing."""
    if image_exists_locally(docker, image):
        return

    con = _stderr_console()
    con.print(f"\n[yellow]⚠  Docker image '{image}' not found locally.[/yellow]")
    con.print("   Benchmark images are typically large (tens of GB).")
    con.print("   This may take a while and use significant disk space.\n")

    if not auto_yes:
        if not sys.stdin.isatty():
            con.print("[red]ERROR: Cannot confirm in non-interactive mode. Use --yes to skip confirmation.[/red]")
            sys.exit(1)
        answer = input("Proceed with docker pull? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            con.print("Aborted.")
            sys.exit(0)

    con.print(f"Pulling {image} ...")
    ret = subprocess.call([docker, "pull", image])
    if ret != 0:
        con.print(f"[red]ERROR: docker pull failed (exit code {ret}).[/red]")
        sys.exit(1)
