# Installation

## Set RobotPkg in your APT system

Ensure you have some required installation dependencies

```bash
sudo apt install -qqy lsb-release gnupg2 curl
```

Add robotpkg as source repository to apt:

```bash
echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list
```
Register the authentication certificate of robotpkg:

```bash
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
```

You need to run at least once apt update to fetch the package descriptions:

```bash
sudo apt-get update
```

## Install the basic dependancies

```bash
# Adapt your desired python version here
sudo apt build-dep robotpkg-py38-pinocchio
sudo apt install robotpkg-py38-casadi robotpkg-py38-eigenpy python3-pip
```

Install python dependancies
```bash
python3 -m pip install --upgrade --user pip
python3 -m pip install --upgrade --user meshcat jupyterlab
```

## Configure your environment

All the packages will be installed in the /opt/openrobots directory. To make use of installed libraries and programs, you must need to configure your PATH, PKG_CONFIG_PATH, PYTHONPATH and other similar environment variables to point inside this directory. For instance:

```bash
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH # Adapt your desired python version here
export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH
```

You may directly add those lines to your $HOME/.bashrc for a persistent configuration.

## Pinocchio

Download and compile pinocchio with the branch pinocchio3-preview.

## Opencv and apriltag

Install python bindings for both libs. This will also install the `.so` libs themselves. 

WARNING! If you have other versions pre-installed, uninstall them before or use python environments to put them out of the way. This is important especially for `apriltag`, since the version installed here is rather old, and you might have a newer one.

```bash
python3 -m pip install opencv-python apriltag
```
## Meshcat

