Overview
======== 

Utilities for emulation of HACC data products. Currently this includes the pre-processing stage of data extraction including read-in, smoothing, scaling, and scale-extension modules.

Installation
============

Start by git cloning this repository and then use


``pip install .``

there are currently few dependencies

Running
=======

The basic set-up of this is that for each type of data there will be a class structure. At the moment the example is the power spectrum which can be initialized by
.. code-block:: python

    from emulator_utils.power_class import PowerSpectrum 
    powerspec = PowerSpectrum()

and then you can read in data to this with 
.. code-block:: python

    powerspec.file_list(path_to_files)
    powerspec.steps()
    powerspec.set_data()

and so on. We'll automate all this in future, but you get the jist.
