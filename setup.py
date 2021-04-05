import os
import setuptools

RESDIR="./results/"
GCNINTERPRET="./gcn_interpret_kmers/"
MODELDIR="./saved_models/"

if not os.path.exists(RESDIR):
    os.makedirs(RESDIR)
    
if not os.path.exists(GCNINTERPRET):
    os.makedirs(GCNINTERPRET)
    
if not os.path.exists(MODELDIR):
    os.makedirs(MODELDIR)
    
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


required = ['keras==2.3.1',
            'numpy==1.16.1',
            'boto3==1.9.239',
            'sklearn',
            'scipy==1.3.2',
            'tqdm==4.38.0',
            'matplotlib==3.0.3',
            'multiprocess']

setuptools.setup(
    name="Nanopore De-novo Modification Inference (NDMI)",
    version="0.0.1",
    author="Ioannis Anastopoulos",
    author_email="ianastop@ucsc.edu",
    description="Repository for reproducing results of manuscript",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ioannisa92/Nanopore_modification_inference",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

