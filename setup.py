import os
import setuptools
import subprocess


def _parse_requirements(requirements_txt_path):
    with open(requirements_txt_path) as fp:
        return fp.read().splitlines()


extras_require = {
    "dev": [
        "pre-commit>=2.0.1",
        "black>=19.10b0",
        "flake8>=3.7",
        "flake8-bugbear>=20.1",
        "sphinx==4.0.2",
        "sphinx-rtd-theme==1.0.0",
        "myst-parser==0.15.1",
        "nbsphinx==0.8.6",
    ],
}


if __name__ == "__main__":
    with open("README.md") as f:
        long_description = f.read()

    cwd = os.path.dirname(os.path.abspath(__file__))
    version = open("version.txt", "r").read().strip()
    sha = "Unknown"

    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        pass

    if sha != "Unknown" and not os.getenv("PAX_RELEASE_BUILD"):
        version += "+" + sha[:7]

    version_path = os.path.join(cwd, "pax", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))

    setuptools.setup(
        name="pax-rl",
        version=version,
        description="Pax: Environment for ...",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Akbir Khan",
        url="https://github.com/akbir/pax",
        license="Apache License, Version 2.0",
        packages=["pax"],
        python_requires=">=3.9",
        install_requires=_parse_requirements("requirements.txt"),
        extras_require=extras_require,
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Development Status :: 4 - Beta",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Games/Entertainment",
        ],
    )
