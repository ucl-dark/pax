# Installation

Pax is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, Haiku does not list JAX as a dependency in requirements.txt.

First, follow these instructions to [install](https://github.com/google/jax#installation) JAX with the relevant accelerator support.

Then, install the following [requirements](https://github.com/ucl-dark/pax/blob/main/requirements.txt).

If you run into into the following problem

```
Traceback (most recent call last):
  File "/usr/lib/python3.9/runpy.py", line 188, in _run_module_as_main
    mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
  File "/usr/lib/python3.9/runpy.py", line 111, in _get_module_details
    __import__(pkg_name)
  File "/home/duser/pax/pax/__init__.py", line 1, in <module>
    from .version import __version__
ModuleNotFoundError: No module named 'pax.version'
```

run `python setup.py sdist`.


