1. Install and configure the GPU drivers (recommended)

- Install the CUDA driver and the CUDA Toolkit.

- Add the ‘lib’ subdirectory (and/or ‘lib64’ subdirectory if you have a 64-bit OS) to your $LD_LIBRARY_PATH environment variable.Set Theano’s config flags

- To use the GPU you need to define the cuda root. You can do it in one of the following ways:

```
Define a $CUDA_ROOT environment variable to equal the cuda root directory, as in CUDA_ROOT=/path/to/cuda/root, 
or add a cuda.root flag to THEANO_FLAGS, as in THEANO_FLAGS='cuda.root=/path/to/cuda/root', 
or add a [cuda] section to your .theanorc file containing the option root = /path/to/cuda/root.
```

2.conda install pygpu==0.7
3.pip install  --user --no-deps  git+https://github.com/Theano/Theano.git#egg=Theano
4.edit ~/.theanorc
```
[global]  
device=gpu  
floatX=float32 

[dnn.conv]
algo_bwd_filter = deterministic
algo_bwd_data = deterministic

[cuda]
root=/usr/local/cuda-9.0

[lib]
cnmem=0.3

[nvcc]
fastmath = True
optimizer_including=cudnn
```
5.conda install Lasagne
