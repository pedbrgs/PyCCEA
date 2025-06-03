import os

from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
if os.path.exists(requirements_path):
    with open(requirements_path, encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []


setup(
    name="pyccea",
    version="1.0.0",
    description="Cooperative co-evolutionary algorithms for feature selection in high-dimensional data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pedro Vinícius A. B. Venâncio",
    author_email="pedbrgs@gmail.com",
    url="https://github.com/pedbrgs/PyCCEA",
    packages=find_packages(include=["pyccea*"]),
    package_data={
        "pyccea": ["datasets/*.parquet", "parameters/*.toml"],
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
