"""Run all compound prefix predictions and the fixed future-image veto."""

from shadow_motion_common import shadow_pipeline


def main():
    import probe_keyframe_rgb_motion as producer

    with shadow_pipeline(producer):
        producer.main()


if __name__ == "__main__":
    main()
