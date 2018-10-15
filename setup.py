import setuptools


setuptools.setup(
    name="pytorch-pcen",
    version="0.0.1",
    author="Ralph Tang",
    author_email="r33tang@uwaterloo.ca",
    description="Efficient implementation of per-channel energy normalization.",
    install_requires=["numpy", "pytorch"],
    url="https://github.com/daemon/pytorch-pcen",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
