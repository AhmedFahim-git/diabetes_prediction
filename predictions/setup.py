from __future__ import annotations

from setuptools import setup

setup(
    name="predictions",
    version="1.0",
    description="A useful module",
    packages=["predictions"],  # same as name
    install_requires=[
        "requests",
        "scikit-learn",
        "pandas",
    ],  # external packages as dependencies
)
