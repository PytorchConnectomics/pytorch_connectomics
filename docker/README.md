# PyTC Docker Guidance

To improve usability, we pushed a PyTC Docker image to the public docker registry.
Additionally, we provide the corresponding Dockerfile to enable individual modifications.

## Prerequisite

- Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart)

### Limitations 

Nvidia-docker is only compatible with Linux distributions. If you are trying to run PyTC Docker on a macOS or Windows machine, please adapt the Dockerfile accordingly and build a new docker image as explained below.  

### Quick setup (03/11/2022)

If your current system does not meet the prerequisite, here is a quick setup guide with the steps copied directly from the official installation websites.

**Docker-CE**

Docker-CE on Ubuntu can be setup using [Docker’s official convenience script](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script):

```bash
$ curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

**NVIDIA Docker**

- Setup the stable repository and the GPG key:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

- Install the nvidia-docker2 package (and dependencies) after updating the package listing:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

- Restart the Docker daemon to complete the installation after setting the default runtime:

```bash
sudo systemctl restart docker
```

- At this point, a working setup can be tested by running a base CUDA container:

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Obtaining a Docker image

To obtain the docker image, pull the prebuilt image from the public registry or build it directly using the provided Dockerfile.

### Building image from Dockerfile

Download the [Dockerfile](Dockerfile) from this directory and run 

```bash
docker build -t [target_tag] <path to folder containing the downloaded Dockerfile>
```

- Replace `[target_tag]` with the name that you want to assign to the image.
- Use `.` for the path argument when running the command inside the directory that contains the Dockerfile.

### Pull pre-built image 

Run the following command to pull the pre-built docker image from the public registry.

```bash
docker pull lauenburg/pytc
```

## Create and run a Docker container


Start an interactive docker session:

```bash
nvidia-docker run -it -p 6006:6006  [target_tag]
```

- If you pulled the image from the public container registry, replace `[target_tag]` with `lauenburg/pytc`.
- We map the container's TCP port `6006` to the port `6006` on the Docker host to access the TensorBoard visualization on the host machine.



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

It is impossible to install or use Docker on the HPC environment, since Docker requires root (`sudo`) privileges.

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