# TODO
try:
    from .algorithm_adapter import BaseAlgorithm, DeploymentMatrix
except ImportError:
    # if run as "__main__"
    from algorithm_adapter import BaseAlgorithm, DeploymentMatrix, abstract_main


__all__ = [
    "BeesColony",
]


class BeesColony(BaseAlgorithm):
    pass


if __name__ == "__main__":
    print("Not implemented yet")
    abstract_main(BeesColony)
