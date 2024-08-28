"""Sets up package for pip installation."""

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="attack",
    version="1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=requirements,
    python_requires=">=3.9.6",
)
