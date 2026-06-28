conda create --name ecotaxa_ml python=3.8
conda activate ecotaxa_ml
conda install tensorflow=2.6.0 scikit-learn=1.0 pandas=1.3.3 imgaug=0.4.0 tensorflow-hub==0.12.0
conda install tqdm
pip install git+https://github.com/ecotaxa/ecotaxa_py_client.git
pip install tensorflow_addons==0.14.0
