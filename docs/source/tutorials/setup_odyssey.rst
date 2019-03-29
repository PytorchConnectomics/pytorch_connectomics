
Guidance for Harvard RC Users
=======================

.. contents::
    :local:

Step 1: `School-related Setup <https://docs.google.com/document/d/18ovdpC2Tzf_8EvDBHXRBznELUbuIQwDN7Llgp_-QbrI/edit>`_
-----------------------

Step 2: Machine Setup
-----------------------
#. Harvard RC server 
  #. Create account:
    * Apply for SEAS account 
    * RC account [[link]](https://www.rc.fas.harvard.edu/resources/access-and-login/)
    * coxfs01 access [[link]](https://portal.rc.fas.harvard.edu/login/?next=/request/grants/add%3Fsearch%3Dcox_lab)

  #. Mount coxfs01 file system to local machine
    * Install packages: `sudo apt-get install cifs-utils`
    * Get your gid on your local machine: `id`
    * Mount it with your rc username and local machine gid: 

    .. code-block:: none

            $ sudo mount -t cifs -o vers=1.0,workgroup=rc,username=${1},gid=${2} //coxfs01.rc.fas.harvard.edu/coxfs01 /mnt/coxfs01
            
  #. Submit jobs through slurm scheduler [[official tutorial]](https://www.rc.fas.harvard.edu/resources/running-jobs/)
  * Get an interactive shell for debug: (${1}: memory in MB, ${2}: # of CPUs, ${3}: # of GPUs)
    + CPU: 
    .. code-block:: none

            $ srun --pty -p cox -t 7-00:00 --mem ${1} -n ${2} /bin/bash

    + GPU: 

    .. code-block:: none

            $ srun --pty -p cox -t 7-00:00 --mem ${1} -n ${2} --gres=gpu:${3} /bin/bash

  * Submit job in the background:
    + `/n/coxfs01/donglai/ppl/public/example_slurm.py`
- Setup CUDA env
  * Request a GPU machine (`srun` or `sbatch`)
  * Load cuda on rc cluster: `module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01`
- Deep learning env (python3/EM-network): `source /n/coxfs01/donglai/lib/miniconda2/bin/activate em-net`
- ssh tunnel for port forwarding (e.g. tensorboard display)
  * Parameters:
    + P1:port you want to display on localhost
    + P2: port on rc server
    + M1: coxgpu name, e.g. coxgpu06
  * On local machine: `ssh -L p1:localhost:p2 xx@login.rc.fas.harvard.edu`
  * On rc login server: `ssh -L p2:localhost:p2 M1`

2. Group server (hp03 machine)
- Get account and IP address: ask Admin
- ssh: `ssh ${IP}`
- Jupyter notebook: `http://${IP}:9999`
- install miniconda
  * local copy (py27): `sh /home/donglai/Downloads/Miniconda2-latest-Linux-x86_64.sh`
  * download [[link]](https://conda.io/en/latest/miniconda.html)
- cmds for neuroglancer
  ```
  screen
  source /home/donglai/miniconda2/bin/activate ng
  python -i xxx.py
  ```

Step 3: Common Practice
-----------------------

- Communication: Slack
- Coding
  * local machine: local development
  * rc server: run big jobs
  * hp03 server: public visualization (html, neuroglancer)
- Project managment
  * Create a new conda env for each project
- Unix Tips
  * Terminal (split screen)
    + On mac: try `iterm2`
    + On Linux: try `terminator` or `tmux`
  * ssh
    + Automatic login in new bashes (after the login in a bash)
      - Create a file with the following content: `vim ~/.ssh/config`
        ```
        Host *
          ControlMaster auto
          ControlPath ~/.ssh/master-%r@%h:%p
        ```
  * bash	
    + Add useful alias: `vim ~/.bashrc`
      ```
      alias csh='ssh ${USERNAME}@login.rc.fas.harvard.edu'
      ```

Step 4: End-to-End Connectomics Tutorial:
-----------------------
- 3D Data visualization with [Neuroglancer](https://github.com/google/neuroglancer)
   * If using jupyter notebook, copy over the kernel folder and choose the kernel `ng`:
   ```
   sudo cp -r /home/donglai/.local/share/jupyter/kernels/ /home/${USERNAME}/.local/share/jupyter/
   ```
   * If using bash, source activate the env: 
   ```
   source /home/donglai/miniconda2/bin/activate ng
   ```
   * Example code on hp03
   ```
   cp /home/donglai/public/tutorial/ng.py ~/
   ```
   * Neuroglancer [shortcuts](https://github.com/google/neuroglancer#keyboard-and-mouse-bindings)
- Image -> Image: deflicker
 * Installation: [[github repo]](https://github.com/donglaiw/EM-preprocess)
 * Run example code: `python script/T_deflicker.py`
- Image -> Affinity: Volumetric Deep learning package
 * Installation: [[github repo]](https://github.com/donglaiw/EM-network)
 * Tensorboard on hp03
   + Activate env: `source /home/donglai/miniconda2/bin/activate tensorB`
   + Run tensorboard (choose an unused port): `tensorboard --logdir=xx --port=10021` 

- Affinity -> segmentation: zwatershed+waterz
  * Paper: [waterz](https://arxiv.org/pdf/1709.02974.pdf), [zwatershed](https://arxiv.org/abs/1505.00249)
  * Installation (github repos): [zwatershed](https://github.com/donglaiw/zwatershed), [waterz](https://github.com/donglaiw/waterz), [evaluation](https://github.com/donglaiw/em-seglib)
  * Example code (on hp03): `cp /home/public/tutorial/*  ~/`
