Neuroglancer
===============

Introduction
-----------------
`Neuroglancer <https://github.com/google/neuroglancer>`_ is a high-performance, flexible WebGL-based viewer and visualization 
framework for volumetric data developed `Google Connectomics Team <https://research.google/teams/connectomics/>`_.
It supports a wide variety of data sources and can display arbitrary (non axis-aligned) cross-sectional views of volumetric 
data and 3-D meshes and line-segment-based models (skeletons). Neuroglancer is a powerful tool for large-scale neuroscience 
datasets, which can be impractical to visualize with other traditional image viewer applications.


Installation and Quick Start
-----------------------------

1 - Install neuroglancer a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two installation instructions are provided. Installing neuroglancer via Python's package manager ``pip`` is simpler. 
If changes to the neuroglancer package or a certain neuroglancer repository is needed, installtion instructions 
for building neuroglancer from source are also provided.

In both cases the software is installed in a Python virtual environment. We recommend to use Anaconda. See
this `page <../notes/installation.html>`_ for steps to create a virtual environment called ``py3_torch``.

.. code-block:: bash 

    # Install neuroglancer using pip in virtual env, which
    # is the recommended way to start quickly.
    source activate py3_torch 
    pip install --upgrade pip
    pip install neuroglancer imageio h5py cloud-volume
    pip install jupyter #(optional) jupyter/ipykernel installation

.. code-block:: bash 

    # build neuroglancer from source (requires nvm/node.js)
    mkdir project
    cd project
    source activate py3_torch

    git clone https://github.com/google/neuroglancer.git
    cd neuroglancer
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")" \
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
    pip install numpy Pillow requests tornado sockjs-tornado six google-apitools selenium imageio h5py cloud-volume
    python setup.py install 


2 - Start a local neuroglancer server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new (initially empty) viewer. This starts a web server in a background thread, which serves a copy of the Neuroglancer 
client, and which also can serve local volume data and handles sending and receiving Neuroglancer state updates and print a link 
to the viewer (only while the script is running). Note that anyone with the link can obtain any authentication credentials that 
the neuroglancer Python module obtains when the viewer is running.

.. code-block:: python

    import neuroglancer

    ip = 'localhost' # or public IP of the machine for sharable display
    port = 9999 # change to an unused port number
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)

    viewer = neuroglancer.Viewer()
    print(viewer)   

.. note::
    Users need to start a local neuroglancer server with ``python -i [YOUR_SCRIPT].py`` or use a jupyter notebook. 
    It cannot be run as a non-interactive python script, *i.e.*, do **not** use ``python [YOUR_SCRIPT].py`` because 
    the server will shut down immediately after running the code.

Publicly available datasets can be loaded either by navigating to the source tab using the GUI or by using the Python interface. Below
is an example to load a public dataset in Python:

.. code-block:: python

    import neuroglancer

    ip = 'localhost' #or public IP of the machine for sharable display
    port = 9999 #change to an unused port number
    neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)

    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        s.layers['image'] = neuroglancer.ImageLayer(source='precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg/')
        s.layers['segmentation'] = neuroglancer.SegmentationLayer(source='precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation', selected_alpha=0.3)

    print(viewer)

Then copy the printed viewer address to your browser (Chrome) to visualize the data.


3 - Using neuroglancer with a local dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The local dataset can be TIFF or HDF5 formats. In this example we use the `SNEMI <../tutorials/neuron.html>`_ neuron 
segmentation dataset for demonstration.

.. code-block:: python

    import neuroglancer
    import numpy as np
    import imageio
    import h5py

    ip = 'localhost' #or public IP of the machine for sharable display
    port = 9999 #change to an unused port number
    neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
    viewer=neuroglancer.Viewer()

    # SNEMI (# 3d vol dim: z,y,x)
    D0='./'
    res = neuroglancer.CoordinateSpace(
            names=['z', 'y', 'x'],
            units=['nm', 'nm', 'nm'],
            scales=[30, 6, 6])

    print('load im and gt segmentation') 
    im = imageio.volread(D0+'train-input.tif')
    with h5py.File(D0+'train_label.h5', 'r') as fl:
        gt = np.array(fl['main'])
    print(im.shape, gt.shape)

    def ngLayer(data,res,oo=[0,0,0],tt='segmentation'):
        return neuroglancer.LocalVolume(data,dimensions=res,volume_type=tt,voxel_offset=oo)

    with viewer.txn() as s:
        s.layers.append(name='im',layer=ngLayer(im,res,tt='image'))
        s.layers.append(name='gt',layer=ngLayer(gt,res,tt='segmentation'))

    print(viewer)

Please note that the mask volume needs to be loaded as a ``'segmentation'`` layer.

4 - Loading public datasets in GUI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different datasets are added sequentially. Use the (+) icon located in the upper left corner to add a new layer. It is designed 
to easily support many different data sources as shown in the image below.  We have to select a data source and enter the 
URL to the data and the layer will be loaded automatically.

.. image :: ../_static/img/new_layer2.png
   :scale: 45%

After adding the source we have to select the **type** of the layer that is loaded. Click on the **new** button and 
select the type of the layer. List of supported data formats are listed `here <https://github.com/google/neuroglancer#supported-data-sources>`_.


Basic usage
--------------
This section shows some basic manipulation instructions that will be useful while viewing a dataset in 
neuroglancer. Here we load the public `FlyEM Hemibrain <https://www.janelia.org/project-team/flyem/hemibrain>`_ dataset 
as an example. In the **top left** corner of the window:

.. image :: ../_static/img/top_left_corner2.png
   :scale: 55%

* The x/y/z denotes the coordinates of the center of the images displayed in 3D space. In this example, the coordinates are (17213, 19862, 20697).
* The numbers inside the parentheses show the resolution of the dataset, in this case each voxel is 8nm x 8nm x 8nm.
* The current coordinates of the cursor are displayed in orange and are continously updated as the position of the cursor changes. In this image the cordinates are (17263, 19919, 29697).

You can load and view multiple layers at once:

.. image :: ../_static/img/screen_cropped2.png

Currently we have two layers loaded

* The image layer (raw images)
* The segmentataion layer (segmentation masks)

The two different tabs marked in the image shown above represent the loaded layers. We can switch them on and off by (left) clicking on their respective names.


You can view all three orthogonal views simultaneously in diffrent frames. There is also an additional frame where we can see the 3D meshes. The three frames and model move together in unison. If you make changes in any of the frames (e.g. rotation, 2D/3D translation), the corresponding changes will be updated in all the projections/models.
You can also change the view of the screen by clicking on top right corner of any of the 3 frames.

.. image :: ../_static/img/screen_VIEWS.png

You can (right) click on the layer tab to display its properties panel:

.. image :: ../_static/img/layer_properties2.png
   :scale: 50%

The graphical rendering can be changed depending on what the layer contains in the rendering tab. The segmentation 
tab (**Seg.**) appears if the layer is a segmentation: 

.. image :: ../_static/img/segmentation_tab2.png
   :scale: 50%

The bottom half displays all the segment names with their corresponding colors and IDs. 
The current active segments are also marked.
The active segments will be visible in the image and 3D mesh view. A single segment can be activated by either double clicking it or by selecting it from the list in the bottom half of the segmentation tab in the properties pane. We can change the opacity and saturation of the selected/non-selected segments from the render tab.
We can also search for a particular segment name, ID or a /regexp using the search bar at the top of the segment pane.
Selecting a single segment shows the segment on the orthagonal frames in its respective color and also renders a 3D mesh.

Some other commonly used commands include:

* zooming in/out (cltr + mousewheel)
* scrolling through the planes (mousewheel)
* selecting a segment (double click)
* snapping back to initial position ('z' key)
* translating (left click and drag)

**The above and other available commands** can be seen in the help menu which can be accessed by pressing **'h'** key.


Loading a mesh layer 
----------------------

.. code-block:: python

    import neuroglancer

    ip = 'localhost' #or public IP of the machine for sharable display
    port = 9999 #change to an unused port number
    neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)

    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        s.layers['image'] = neuroglancer.ImageLayer(source='precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe')
        s.layers['mesh'] = neuroglancer.SingleMeshLayer(source='vtk://https://storage.googleapis.com/neuroglancer-fafb-data/elmr-data/FAFB.surf.vtk.gz')

    print(viewer)


Showing array of active segments
----------------------------------------

This code outputs the currently selected layers. The code can be added to a python script or run as a python notebook codeblock.

.. code-block:: python

    # assume a viewer with a 'segmentation' layer is created
    import numpy as np
    import time        

    while True:
        print(np.array(list(viewer.state.layers['segmentation'].segments)))
        time.sleep(3)

Logging mouse position and selected layers
--------------------------------------------

This code can be used to log (output in terminal) the current mouse position in voxel space and the selected layers. 
A log is created if the key ``L`` is pressed. The code can be added to a python script or run as a python notebook codeblock.

.. code-block:: python

    # assume a viewer with is already created
    import numpy as np

    num_actions = 0
    def logger(s):
        global num_actions
        num_actions += 1
        with viewer.config_state.txn() as st:
            st.status_messages['hello'] = ('Got action %d: mouse position = %r' %
                                        (num_actions, s.mouse_voxel_coordinates))

        print('Log event')
        print('Mouse position: ', np.array(s.mouse_voxel_coordinates))
        print('Layer selected values:', (np.array(list(viewer.state.layers['segmentation'].segments))))


    viewer.actions.add('logger', logger)
    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer['keyl'] = 'logger'
        s.status_messages['hello'] = 'Add a promt for neuroglancer'
