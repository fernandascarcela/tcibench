__all__ = [
    "balance_data",
    "generate_with_synthcity",
    "check_artifact",
]


def __getattr__(name):
    """Load each baseline implementation only when it is requested."""
    if name == "balance_data":
        from .classical_baselines import balance_data

        return balance_data
    if name == "check_artifact":
        from .classical_baselines import check_artifact

        return check_artifact
    if name == "generate_with_synthcity":
        from .generative_baselines import generate_with_synthcity

        return generate_with_synthcity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
