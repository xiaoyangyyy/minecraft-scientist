"""Capture source-bound distinct-handle compounds at untouched natural spawn."""

from __future__ import annotations

from pathlib import Path

from minecraft_scientist.exploration.compound_motion import compound_bindings


def main():
    import capture_controlled_rgb_motion as capture

    # The canonical CLI validates seed, image size, time and weather as before.
    # Compound defaults/timeline/pairs and this wrapper are frozen before launch.
    with compound_bindings(
        capture, source_function="source_paths", wrapper=Path(__file__)
    ):
        capture.main()


if __name__ == "__main__":
    main()
