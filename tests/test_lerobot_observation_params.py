"""LeRobot bridge observation params: ``quat_no_antipodal`` is opt-in and reaches the LIBERO constructor."""

from __future__ import annotations

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.model_servers import lerobot as lr
from vla_eval.orchestrator import _merge_observation_params

TWO_CAMERAS = ["observation.images.camera1", "observation.images.camera2"]


def _server(quat_no_antipodal: bool) -> lr.LeRobotModelServer:
    """``LeRobotModelServer`` without ``__init__``: only what ``get_observation_params`` reads."""
    server = lr.LeRobotModelServer.__new__(lr.LeRobotModelServer)
    server.state_key = "observation.state"
    server._expected_image_keys = TWO_CAMERAS
    server._image_keys_arg = None
    server._quat_no_antipodal = quat_no_antipodal
    return server


def test_default_quat_no_antipodal_leaves_the_params_dict_unchanged() -> None:
    """Off by default: the params are exactly what the bridge sent before this arg existed."""
    params = _server(quat_no_antipodal=False).get_observation_params()

    assert params == {"send_state": True, "send_wrist_image": True}


def test_quat_no_antipodal_reaches_libero_constructor() -> None:
    """Enabled on the server, it is forwarded to a parameter LIBEROBenchmark accepts."""
    params = _server(quat_no_antipodal=True).get_observation_params()

    merged = _merge_observation_params(LIBEROBenchmark, {}, params)

    assert merged["quat_no_antipodal"] is True


def test_eval_config_overrides_server_quat_no_antipodal() -> None:
    """An explicit ``quat_no_antipodal: false`` in the eval config wins over the server."""
    params = _server(quat_no_antipodal=True).get_observation_params()

    merged = _merge_observation_params(LIBEROBenchmark, {"quat_no_antipodal": False}, params)

    assert merged["quat_no_antipodal"] is False
