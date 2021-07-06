import setuptools

with open("LICENSE", "r") as fh:
    license = fh.read()

with open("README.md", "r") as fh:
    long_description = fh.read()

short_description = (
    "This package provides a collection of modules and examples to quickly "
    "setup experiments using Dual SVMPC."
)
setuptools.setup(
    name="dust",
    version="0.0.1",
    author="Lucas Barcelos",
    author_email="lucas.barcelos@sydney.edu.au",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lubaroli/dust",
    packages=setuptools.find_packages(exclude=("notebooks", "docs", "scripts")),
    license=license,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.8.1",
        "gpytorch >= 1.3",
        "gym >= 0.18",
        "pandas",
        "scipy",
        "optuna",
        "KDEpy",
        "tqdm",
        "pyyaml",
        "moviepy",
        "altair",
        "seaborn",
        "matplotlib",
        "dill",
    ],
)
