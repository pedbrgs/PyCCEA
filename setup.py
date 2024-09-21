from setuptools import setup, find_packages


with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines()]


setup(
    name="pyccea",
    version="0.1.0",
    description="Cooperative co-evolutionary algorithms for feature selection in big data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pedro Vinícius Almeida Borges de Venâncio",
    author_email="pedbrgs@gmail.com",
    url="https://github.com/pedbrgs/PyCCEA",
    packages=find_packages(include=["pyccea*"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
