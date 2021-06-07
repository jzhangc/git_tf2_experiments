# Apple tensorflow2.4 with gpu acceleration

experimental codes for apple tensorflow2.4 with gpu acceleration

## Install via conda enviroment

1. (Apple and Intel chips) install miniforge3 (mini coda with forge channel as default)

   url: <https://github.com/conda-forge/miniforge#miniforge3>

        bash Miniforge3-MacOSX-arm64.sh

   NOTE: Although it is possible to use `conda config --add channels conda-forge`  to manually add the `forge` channel to miniconda, it is generally recommended using miniforge version of conda for both Apple and Intel chips. 

2. create a conda environment and install via the yml file

        conda env create -f ./inst/environment.yml --prefix ./conda_venv_atf24

3. install additional pacakges

        conda activate ./conda_venv_atf24
        conda install scikit-learn setuptools cached-property six packaging matplotlib autopep8 jupyter tqdm pandas

   If with manually added forge channel, use the following:

        conda install scikit-learn setuptools cached-property six packaging matplotlib autopep8 jupyter tqdm pandas -c conda-forge

4. install apple tf2.4

    Intel chips:

        pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_x86_64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_x86_64.whl 

    NOTE: run "export SYSTEM_VERSION_COMPAT=0" if "not supported" error is returned

    Apple chips:

        pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl

## Known issues

    - ATF24 does not work with numpy version newer than 1.19.5
    - evaluate and predict functions conflict (cannot use validation data in model.fit)