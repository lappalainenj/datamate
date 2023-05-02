from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_version():
    from pathlib import Path

    root = Path(__file__).parent
    string = open(root / "datamate/version.py", "r").read().strip()
    return string.split("=")[-1].strip().strip('"')


setup(
    name="datamate",
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        "pandas",
        "toolz",
        "numpy",
        "typing_extensions",
        "h5py>=3.6.0",
        "ipython<8.5",  # cause of https://github.com/ipython/ipython/issues/13830
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
