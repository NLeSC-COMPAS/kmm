Installation
============

To install KMM first clone the repository, including the submodules, with the following command:

.. code-block:: shell

    git clone --recurse-submodules https://github.com/NLeSC-COMPAS/kmm.git

After cloning the repository on your machine, you can build KMM using ``cmake`` and ``make``:

.. code-block:: shell

    cd kmm/build
    cmake ..
    make

There are few CMake options that users can set. First, it is possible to control the backend used to run GPU code with:

* ``KMM_USE_CUDA``: to enable CUDA support in KMM
* ``KMM_USE_HIP``: to enable HIP support in KMM

The two options are mutually exclusive; if neither is set, a dummy backend is used instead.
The next option is used to control how the library is compiled:

* ``KMM_STATIC``: to build KMM as a static library

The default behavior is to build KMM as a dynamic library.
The remaining options are aimed at KMM developers:

* ``KMM_ENABLE_LINTER``: to run a linter on the C++ code
* ``KMM_BUILD_TESTS``: to build the tests
* ``KMM_BUILD_EXAMPLES``: to build the examples
* ``KMM_BUILD_BENCHMARKS``: to build the benchmarks

If not specified, the linter is not used, and tests, examples, and benchmarks are not compiled.

