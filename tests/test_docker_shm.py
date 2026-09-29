"""``render: cpu`` Docker runs get a usable /dev/shm (lavapipe keeps device memory there)."""

from vla_eval.cli._docker import build_docker_command


def _cmd(config):
    return build_docker_command("docker", config, "/out", "/cfg.yaml", "c", interactive=False)


def test_cpu_render_sets_shm_size():
    cmd = _cmd({"docker": {"image": "img"}, "render": "cpu"})
    assert cmd[cmd.index("--shm-size") + 1] == "16g"
    cmd = _cmd({"docker": {"image": "img", "shm_size": "2g"}, "render": " CPU "})
    assert cmd[cmd.index("--shm-size") + 1] == "2g"


def test_gpu_render_leaves_docker_default():
    assert "--shm-size" not in _cmd({"docker": {"image": "img"}})
