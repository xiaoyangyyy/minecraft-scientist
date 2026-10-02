"""Infer all prefixes from a publicly verified prospective active acquisition."""

from active_motion_pipeline import active_pipeline


def main():
    import probe_keyframe_rgb_motion as producer

    with active_pipeline(producer):
        producer.main()


if __name__ == "__main__":
    main()
