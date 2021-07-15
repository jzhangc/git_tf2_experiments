# Tensorflow 2.5 with Apple-Metal gpu acceleration

Experimental codes for Tensorflow 2.5 with Apple-Metal GPU acceleration for Mac (both Intel and Apple chips)

## UPDATE (June.2021)

Apple has now implmented metal TF2.5 plugin for GPU acceleration. See [here](https://developer.apple.com/metal/tensorflow-plugin/) for details.

1. System requrement

    Apple suggests to have at least macOS 12.0 (Monterey), which has not been out to the public yet.

    However, test seemed to be successful on a M1 13 inch Macbook Pro running macOS 11.4 (Big Sur)

2. Install miniforge3 (mini coda with forge channel as default)

   url: <https://github.com/conda-forge/miniforge#miniforge3>

        bash Miniforge3-MacOSX-arm64.sh

   NOTE 1: Although it is possible to use `conda config --add channels conda-forge`  to manually add the `forge` channel to miniconda, it is generally recommended using miniforge version of conda for both Apple and Intel chips.

3. Create, setup and activate a conda environment

        conda env create -f ./inst/environment_generic.yml --prefix ./conda_venv_tf_metal
        conda activate ./conda_venv_tf_metal

4. (Apple chip) Install tensorflow dependencies

        conda install -c apple tensorflow-deps

5. Install base tensorflowy

        python -m pip install tensorflow-macos

6. Install metal plugin

        python -m pip install tensorflow-metal

7. Install other conda packages

        conda install pandas scikit-learn jupyter matplotlib tqdm autopep8

   Intel only:

        conda install imutils

8. Install other pip package(s)

        python -m pip install scikit-multilearn

## Install via conda enviroment (OLD)

1. (Apple and Intel chips) install miniforge3 (mini coda with forge channel as default)

2. create a conda environment and install via the yml file

        conda env create -f ./inst/environment.yml --prefix ./conda_venv_atf24

3. install additional pacakges

        conda env create -f ./inst/environment.yml --prefix ./conda_venv_atf24
        conda install scikit-learn setuptools cached-property six packaging matplotlib autopep8 jupyter tqdm pandas numpy

   If with manually added forge channel, use the following:

        conda install scikit-learn setuptools cached-property six packaging matplotlib autopep8 jupyter tqdm pandas -c conda-forge

4. install apple tf2.4

    Intel chips:

        pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_x86_64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_x86_64.whl 

    NOTE: run "export SYSTEM_VERSION_COMPAT=0" if "not supported" error is returned

    Apple chips:

        pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl

5. Known issues

    - ATF24 does not work with numpy version newer than 1.19.5
    - evaluate and predict functions conflict (cannot use validation data in model.fit)
