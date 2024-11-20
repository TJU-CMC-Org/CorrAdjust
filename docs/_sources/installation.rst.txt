:html_theme.sidebar_secondary.remove:

Installation
============

CorrAdjust requires Python 3.10-3.12.
Although not mandatory, we strongly recommend installing it
into a separate Conda or virtualenv environment.

On Linux, you can install CorrAdjust from PyPI with a single command:

.. code-block:: bash

   pip install corradjust

You can verify the installation by running the automatic test on synthetic data.
Just run this command from any writable directory:

.. code-block:: bash

   pytest -v -s --pyargs corradjust

On macOS, two extra commands are required:

.. code-block:: bash

   brew install gcc pkg-config tbb
   export CXX=$(brew --prefix gcc)/bin/g++-14
   pip install corradjust

This is to ensure proper compilation for the ``parallel_sort`` module.
Please change ``g++-14`` to the actual version installed on your system.

If you are using Windows, we recommend installing
Windows Subsystem for Linux (WSL) or any alternative
that allows running Linux environment on your machine.