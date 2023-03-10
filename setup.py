from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="datamate",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "toolz",
        "numpy",
        "typing_extensions",
        "h5py>=3.6.0",
        "notebook",
        "ipywidgets",
        "tqdm",
        "matplotlib",
        "ruamel.yaml",
    ],
    author="Janne Lappalainen & Mason McGill",
    description="A data organization and compilation system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
