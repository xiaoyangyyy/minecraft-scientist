"""Compatibility module; shared implementation now lives in the main package."""

import sys

from minecraft_scientist.evaluation.bootstrap import __name__ as _implementation_name

sys.modules[__name__] = sys.modules[_implementation_name]
