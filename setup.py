import os
import setuptools


def _parse_requirements(requirements_txt_path):
    with open(requirements_txt_path) as fp:
        return fp.read().splitlines()


extras_require = {
    "dev": [
        "pre-commit>=2.0.1",
        "black>=19.10b0",
        "flake8>=3.7",
        "flake8-bugbear>=20.1",
    ],
}


if __name__ == "__main__":
    with open("README.md") as f:
        long_description = f.read()

    setuptools.setup(
        name="pax",
        version="0.1.0b",
        description="Pax: Environment for ...",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Akbir Khan",
        url="https://github.com/akbir/pax",
        license="Apache License, Version 2.0",
        packages=["pax"],
        python_requires=">=3.7",
        install_requires=_parse_requirements("requirements.txt"),
        extras_require=extras_require,
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Development Status :: 4 - Beta",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Games/Entertainment",
        ],
    )
