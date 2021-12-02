Neuroglancer (Draft)
================================================

Introduction
-----------------
`Neuroglancer <https://github.com/google/neuroglancer>`_ is a high-performance, flexible WebGL-based viewer and visualization framework for volumetric data.
It supports a wide variety of data sources and is capable of displaying arbitrary (non axis-aligned) cross-sectional views of volumetric data, as well as 3-D meshes and line-segment based models (skeletons).
Neuroglancer is a very useful tool for neuroscience datasets and ones that can be impractical to view in other traditional image viewer applications.


Installation and quickstart instructions
------------------------------------------------

Installation instructions tested on Ubuntu 20.04.


1 - Install an IDE and python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install an integrated development environment (e.g. `Visual Studio Code <https://code.visualstudio.com/Download>`_) and proceed to install python (>3.8).

    .. code-block:: bash

        $ apt update && apt upgrade
        $ apt install python3 

2 - Create a virtual environment and install neuroglancer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two installation instructions are provided. Installing neuroglancer via python's package manager pip is simpler. 
If changes to the neuroglancer package or a certain neuroglancer repository is needed, installtion instructions for building neuroglancer from source are also provided.

In both cases the software is installed in a python virtual environment.



    .. code-block:: bash 

        #install neuroglancer using pip 
        $ python3 -m venv neuroglancer_venv
        $ source neuroglancer_venv/bin/activate
        $ pip3 install --upgrade pip
        $ pip3 install neuroglancer imageio h5py cloud-volume
        $ pip3 install jupyter #(optional) jupyter/ipykernel installation
        $ jupyter notebook #(optional) open jupyter notebook

        #build neuroglancer from source (requires nvm/node.js)
        $ mkdir project
        $ cd project
        $ python3 -m venv neuroglancer_venv
        $ source neuroglancer_venv/bin/activate
        $ git clone https://github.com/google/neuroglancer.git
        $ cd neuroglancer
        $ curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        $ export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
          [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
        $ pip3 install numpy Pillow requests tornado sockjs-tornado six google-apitools selenium imageio h5py cloud-volume
        $ python3 setup.py install 

        #close virtual environment
        $ deactivate

3 - Start a local neuroglancer server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Create a new (initially empty) viewer. This starts a webserver in a background thread, which serves a copy of the Neuroglancer client, and which also can serve local volume data and handles sending and receiving Neuroglancer state updates and print a link to the viewer (only while in script is running). Note that while the Viewer is running, anyone with the link can obtain any authentication credentials that the neuroglancer Python module obtains.

    .. code-block:: python

        import neuroglancer

        ip = 'localhost' #or public IP of the machine for sharable display
        port = 9999 #change to an unused port number
        neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)

        viewer = neuroglancer.Viewer()
        print(viewer)   

    Publicly available datasets can be loaded either by navigating to the source tab (section 4) using the gui or by using the python interface. 

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


 

4 - (Optional) Start neuroglancer with a local dataset (.tif image raster/ .h5 3D volume)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   * A data example is provided, containing an image raster [.tif] and a 3D volume [.h5], otherwise the `SNEMI? <x.com>`_ neuron segmentation dataset or others can be used.
   
   * To start a local neuroglancer server with run the script ``python3 -i THIS_FILE.py`` or as a jupyter notebook. It cannot be run as a non-interactive python script, e.g. do not use python3 THIS_File.py because the server will shut down immediately after running the code.

    .. code-block:: python
    
        import neuroglancer
        import numpy as np
        import imageio


        ip = 'localhost' #or public IP of the machine for sharable display
        port = 9999 #change to an unused port number
        neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
        viewer=neuroglancer.Viewer()

        # SNEMI
        D0='./'
        res = neuroglancer.CoordinateSpace(
                names=['z', 'y', 'x'],
                units=['nm', 'nm', 'nm'],
                scales=[30, 6, 6])

        print('load im and gt seg')
        # 3d vol dim: z,y,x 
        im = imageio.volread(D0+'train-input.tif')
        gt = np.load('data.npy')


        def ngLayer(data,res,oo=[0,0,0],tt='segmentation'):
            return neuroglancer.LocalVolume(data,dimensions=res,volume_type=tt,voxel_offset=oo)

        with viewer.txn() as s:
            s.layers.append(name='im',layer=ngLayer(im,res,tt='image'))
            s.layers.append(name='gt',layer=ngLayer(gt,res, tt='image'))

        print(viewer)


4 - (Optional) Loading public datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different datasets added sequentially. Use the (+) icon located in the upper left corner to add a new layer.


It is designed to easily support many different data sources as shown in the image below. 
We have to select a data source and enter the URL to the data and the layer will be loaded automatically.

.. image :: ../_static/img/new_layer2.png
   :scale: 50%

After adding the source we have to select the **type** of the layer that is loaded. Click on the **new** button and select the type of the layer. 


Add gui use here.


Basic usage
--------------
This section shows some basic manipulation instructions that will be useful while viewing a dataset in neuroglancer.

In the **top left** corner of the window:

.. image :: ../_static/img/top_left_corner2.png
   :scale: 58%

* The x/y/z denotes the coordinates of the center of the images displayed in 3D space. In this example, the coordinates are (17213, 19862, 20697).
* The numbers inside the parentheses show the resolution of the dataset, in this case each voxel is 8nm by 8nm by 8nm.
* The current coordinates of the cursor are displayed in orange and are continously updated as the position of the cursor changes. In this image the cordinates are (17263, 19919, 29697).

You can load and view multiple layers at once:

.. image :: ../_static/img/screen_cropped2.png

Currently we have two layers loaded

* The image layer(image)
* The segmentataion layer(segmentation)

The two different tabs marked in the image shown above represent the loaded layers. We can switch them on and off by (left) clicking on their respective names.


You can view all three orthogonal views simultaneously in diffrent frames. There is also an additional frame where we can see the 3D meshes. The three frames and model move together in unison. If you make changes in any of the frames (e.g. rotation, 2D/3D translation), the corresponding changes will be updated in all the projections/models.
You can also change the view of the screen by clicking on top right corner of any of the 3 frames.

.. image :: ../_static/img/screen_VIEWS.png

You can (right) click on the layer tab to display its properties panel:

.. image :: ../_static/img/layer_properties2.png
   :scale: 50%

The graphical rendering can be changed depending on what the layer contains in the rendering tab.

The segmentation tab appears if the layer is a segmentation: 

.. image :: ../_static/img/segmentation_tab2.png
   :scale: 50%

The bottom half displays all the segment names with their corresponding colors and IDs. 
The current active segments are also marked.
The active segments will be visible in the image and 3D view. A single segment can be activated by either double clicking it or by selecting it from the list in the bottom half of the segmentation tab in the properties pane. We can change the opacity and saturation of the selected/non-selected segments from the render tab.
We can also search for a particular segment name, ID or a /regexp using the search bar at the top of the segment pane.
Selecting a single segment shows the segment on the orthagonal frames in its respective color and also renders a 3D mesh.

Some other common commands include

* zooming in/out (cltr + mousewheel)
* scrolling through the planes (mousewheel)
* selecting a segment (double click)
* snapping back to initial position ('z' key)
* translating (left click and drag)

**These and other commands** can be seen in the help menu which can be accessed by pressing **'h'** key.


(Example) Loading a mesh layer 
--------------------------------

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


(Example) Get array of active segments
----------------------------------------

This code outputs the currently selected layers.

The code can be added to a python script or run as a python notebook codeblock.

    .. code-block:: python

        import numpy as np
        import time        

        while True:
            print(np.array(list(viewer.state.layers['segmentation'].segments)))
            time.sleep(2)

(Example) Log current mouse position and selected layers
------------------------------------------------------------

    This code can be used to log (output in terminal) the current mouse position in voxel space and the selected layers. A log is created if the letter ``l`` is pressed.

    The code can be added to a python script or run as a python notebook codeblock.

    .. code-block:: python

        import numpy as np

        num_actions = 0
        def logger(s):
            global num_actions
            num_actions += 1
            with viewer.config_state.txn() as st:
                st.status_messages['hello'] = ('Got action %d: mouse position = %r' %
                                            (num_actions, s.mouse_voxel_coordinates))

            print('Log event')
            print('  Mouse position: ', np.array(s.mouse_voxel_coordinates))
            print('  Layer selected values:', (np.array(list(viewer.state.layers['segmentation'].segments))))
    
    
        viewer.actions.add('logger', logger)
        with viewer.config_state.txn() as s:
            s.input_event_bindings.viewer['keyl'] = 'logger'
            s.status_messages['hello'] = 'Add a promt for neuroglancer'