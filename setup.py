from setuptools import find_packages, setup

# MILO is importable as the `milo` package (e.g. milo.pipeline.*). The fitting
# engine under milo/fit is a hydra app run with cwd=milo/fit (relative imports),
# so it is not imported as milo.fit.* — find_packages registers it harmlessly.
setup(
    name="milo",
    version="0.1.0",
    description="MILO: Reconstructing Humans and Objects in Interaction using Large Reconstruction Models",
    packages=find_packages(include=["milo", "milo.*"]),
)
