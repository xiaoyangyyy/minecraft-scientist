"""Run the unchanged chronological selector over preregistered compound captures."""

from __future__ import annotations

from compound_motion_common import compound_pipeline


def main():
    import probe_keyframe_rgb_motion as producer

    with compound_pipeline(producer):
        producer.main()


if __name__ == "__main__":
    main()
