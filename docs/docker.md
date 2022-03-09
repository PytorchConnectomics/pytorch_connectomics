# PyCT Docker

We pushed a PyCT Docker image to the public docker registry to improve usability.
Additionally, we provide the corresponding Dockerfile to enable individual modifications.

## Prerequisite

- Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart)

## Limitations 

Nvidia-docker is only compatible with Linux distributions. If you are trying to run PyCT Docker on a macOS or Windows machine, please adapt the Dockerfile accordingly and build a new docker image as explained below.  

## Obtaining a Docker image

### Pull pre-built image 

Run the following command to pull the pre-built docker image from the public registry.

```bash
docker pull lauenburg/pyct_docker
```

### Building image from Dockerfile

Download the [Dockerfile](Dockerfile) from this directory and run 

```bash
docker build -t [target_tag] <path to folder containing the downloaded Dockerfile>
```
Use `.` for the path argument when running the command inside the directory that contains the Dockerfile.

## Create and run a Docker container


Start an interactive docker session:

```bash
nvidia-docker run -it -p 6006:6006  lauenburg/pyct_docker
```

- If you build the docker image from the Dockerfile, replace `lauenburg/pyct_docker` with the `target_tag` defined in the build command
- We map the container's TCP port `6006` to the port `6006` on the Docker host to access the TensorBoard visualization on the host machine.


## Use the PyCT container

- Now you are in the Docker environment. Go to our code repo and start running things.
```bash
cd /workspace/pytorch-CycleGAN-and-pix2pix
bash datasets/download_pix2pix_dataset.sh facades
python -m visdom.server &
bash scripts/train_pix2pix.sh
```

## Hints and additional information

- Keywords

    - Dockerfile: A recipe for creating Docker images
    - Docker image: A read-only template used to build containers
    - Container: A deployed instance created from a Docker image


- Docker commands

    - `docker build`: Build a new image from a Dockerfile
    - `docker create`: Creates a writeable container from an image and prepares it for running.
    - `docker run`: Creates a container (same as Docker create) and runs it.


- Interactive mode

We did not define a command or entry point at the end of our Dockerfile. It is, therefore, necessary to run the docker image in interactive mode. Docker's primary purpose is to create and deploy services. It, therefore, requires a command to keep running in the foreground. Otherwise, it thinks that the application stopped and shuts down the container. We start the container with an active bash as the foreground process when running it in interactive mode.

- HPC users

It is impossible to install or use Docker on the HPC environment, since Docker requires root privileges.

> Docker containers need root privileges for full functionality, which is not suitable for a shared HPC environment. 
>
> <cite>https://docs.rc.fas.harvard.edu/kb/singularity-on-the-cluster/</cite>

However, it is possible to pull and run the pre-build container image from the public registry using Singularity.
Singularity is preinstalled on the cluster. For more information click [here](https://docs.rc.fas.harvard.edu/kb/singularity-on-the-cluster)

- GCP

When creating a GPU compute instance using the Google Cloud Platform, you need to install the Nvidia driver.
The easiest way to do this is using the install script referenced by GCP [here](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#installation_scripts).


- Privilege

You may have to run the docker commands using `sudo`