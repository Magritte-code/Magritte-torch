.. _link-synthetic_observations:


Synthetic observations
######################

The following examples show how to create synthetic observations with Magrittetorch.

.. Note::
    In these examples we assume to have a Magrittetorch model available. For more information
    on how to create a Magrittetorch model, please refer to the :ref:`creating models
    <link-creating_models>` or .. :ref:`post-processing examples <link-post-processing>`.


Most of the following notebooks create synthetic channel maps. These are a series of 2D images that show which part of the model is visible at a given frequency.
Because of the interaction of the Doppler shift with the velocity field, we can observe 
which part of the model is moving away from us (thus redshifted; denoted with positive velocity) 
and which part is moving towards us (thus blueshifted; denoted with negative velocity).

.. toctree::
   :maxdepth: 1
   :caption: Contents
   
   0_image_analytic_disk.ipynb
   1_image_analytic_spiral.ipynb
   2_create_and_image_Phantom.ipynb
   3_image_3D_AMRVAC.ipynb
   4_image_3D_AMRVAC_red.ipynb
   5_image_3D_Phantom.ipynb
   6_image_3D_Phantom_red.ipynb
   7_image_3D_Phantom_red_NLTE.ipynb