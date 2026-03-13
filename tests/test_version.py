"""Smoke test: package is importable and exposes a version string."""


def test_version_is_string():
    from vla_eval import __version__

    assert isinstance(__version__, str)
    assert __version__  # not empty
