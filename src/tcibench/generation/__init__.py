__all__ = [
    "balance_data",
    "generate_with_synthcity",
    "check_artifact",
]


def __getattr__(name):
    """Load optional baseline dependencies only when they are requested."""
    if name in __all__:
        from . import baselines

        return getattr(baselines, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
