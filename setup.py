import os
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

current_dir = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(current_dir, "requirements.txt"), encoding="utf-8") as f:
        install_requires = f.read().split("\n")
except FileNotFoundError:
    install_requires = []


setup(
    name="projkit",
    version="0.1.0",
    author="Fuzail Palnak",
    author_email="fuzailpalnak@gmail.com",
    url="https://github.com/fuzailpalnak/projkit",
    description="Library designed to simplify camera projection tasks and calculations, "
                "particularly when working with image predictions and 3D point cloud data."
                " This library provides functions to effectively incorporate point cloud data with image predictions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
    keywords=[
        "Camera Projections",
        "3d Projections",
        "2d projections",
        "point cloud",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
    ],
)