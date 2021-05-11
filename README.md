# apple tensorflow2.4 with gpu acceleration
experimental codes for apple tensorflow2.4 with gpu acceleration

# install via conda enviroment
1. add conda-forge channel

    conda config --add channels conda-forge nodefaults

2. create a conda environment and install via the yml file

    conda env create -f ./inst/environment.yml --prefix ./conda_venv_atf24

3. install apple tf2.4

    Intel chips:

    pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_x86_64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_x86_64.whl 

    NOTE: run "export SYSTEM_VERSION_COMPAT=0" if not supported error

    Apple chips:

    pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl
