# pc_diff_model
point cloud diffusion model

## Open-source code options:
I considered using one of the three diffusion models below. Eventually I decided to go with number 1, for the moment.

1. The implemented model in this repo is a modified version of the generative diffusion model proposed in "[Diffusion Probabilistic Models for 3D Point Cloud Generation](https://github.com/luost26/diffusion-point-cloud)". This model is a good basic diffusion model for point cloud generation. I added a generative branch based on the [Normalizing Flows](https://proceedings.mlr.press/v37/rezende15) to the model based on the assumtion that the new branch would produce noisy point clouds similar to laser scanners produces. However, there isn't enough time to write the Blender script I'm thinking for generating appropriate data for training this branch properly. The script will load meshes, cut pieces of them and save them to be used when training the generating component of the diffusion model. The challenge here is that the Blender script must be written pretty efficiently and do this process on-the-fly otherwise it would take a very long time. So I have to defer this to the next phase if that's an option. At the moment I just add some random noise to the point clouds when training the generative component to simulate the output of a "noisy" laser scanner. Note that the autoencoder part of the model is a [deterministic] diffusion process and is used for reconstruction purposes only.

I made lots of little changes to the original model. Here are some of the changes I made that I remember. I try to sort them by priority:
- Added a new module/branch to the model for generation instead of training two separate models for reconstruction and generation. This way I can integrate both generative and reconstruction components in a single model and make training much more efficient (probably by a factor of 2.5-3 times). For this, the model uses the the "code" (i.e. encoded point cloud) that the reconstruction branch of the model infers; in other words, gradient flow for updating part of the encoder that calculates the mean comes from reconstruction only. Although at the moment the generated shapes do not resemble anything. I assume this is because of the "wrong" way I injected noise by adding completely random noise to the original point clouds to guide the generation process
- Added Chamfer Distance (still have to resolve some compilation issues to get it to work on GPU) and EMD metrics to improve training in addition to the original MSE loss. These two metrics is extremely computationally intensive so they are called 1% of the times during training. Even with being called 1% of the time, GPU utilization is down by about 50-60% compared to when CD and EMD metric functions are not called
- Made the code more lean and readable
- Made the code less hard to debug by removing hard-coded arguments so a small change in the input or parameters wouldn't cause unexpected crashes
- Using better a optimization method than the original paper
- Added weight initialization


2. I initially intented to use [LION: Latent Point Diffusion Models for 3D Shape Generation](https://research.nvidia.com/labs/toronto-ai/LION/) due to its hierarchical latent structure that allows more efficient learning and better generalization that is needed for our applications but it requires more computational resources than codebase no. 1 (I see that they talk about running on a few GPUs but codebase no. 1 runs on a single GPU without issues) and their codebase  which would take me longer to do debugging. The generative model in this work is more expressive than the codebase no. 1 and does not have a separate reconstruction and generation pipeline which is desirable. Also, I'm not sure if licensing would be an issue if parts of this codebase would end up in an actual product.

3. [Point-E](https://github.com/openai/point-e) would be the best diffusion model out of these three for point clouds: it is more efficient and produces shapes with good quality but I decided to not use it for now as it seems I would need to spend more time to modify it and do debugging. Maybe I can try Point-E later and use the way objects are encoded in LION to improve its performance if this would be an option.

One thing about all of these models is that I'm not sure at the moment that they would eventually be able to do what we want out of the box and even with some modifications to improve them because the objects Machine Labs has are not common objects and have novelty almost all the time which would make it harder for these models to encode and generate samples with good fedelity from. I think what we would eventually need is a compositional diffusion model that can produce and explain novel objects. Also, I think we will potentially benefit from learning a simulator for the laser scanning process and formulate the problem of shape completion/generation as solving an inverse problem.

## Dataset

FOr training used the ShapeNet Core dataset used in codebase no. 1. The authors have sampled ShapeNet Core objects with 2048 points and split the data into train/validation/test sets that I used. You can download the dataset from [here](https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ). I also wanted to try higher number of sampled points on shapes but decided not to due to computation cost. [this](https://github.com/stevenygd/PointFlow?tab=readme-ov-file#dataset) is another ShapeNet dataset sampled with more points in case you want to try it.

## Evaluation

### Metrics
I evaluated the results using the Chamfer Distance (bidirectional) and Earth Mover's Distance metrics in addition to MSE loss that is used for reconstruction. Here is a screenshot of the losses. Note that the total training loss is not intuitive because I am summing up the result of 4 metrics: MSE for reconstruction , MSE for generation (which is somewhat meaningless at the moment due to inavailability of ground-truth good noisy data) in addition to CD and EMD; I should have decoupled theses metrics for visualization purposes but forgot to do it before running the final training jobs. However, you can see graphs for reconstruction (MSE), CD and EMD for test samples has been going down consistently so I assume something similar happens for the training set.

<img src=evaluation/tensorboard_metrics1.png>
<img src=evaluation/tensorboard_metrics2.png>
<img src=evaluation/tensorboard_metrics3.png>

### Qualitative
The reconstructions are not bad given that the model has only about 2.7 million parameters and has been trained on all objects from all categories. I think I could get much better reconstruction quality if I had used one category (`args.category='airplane` etc) during training. The generated samples are pretty bad; I think this is most likely due to some bug in the code or it could be due to the way I have coupled both the generative and reconstruction model into one model. I didn't have enough time to debug the code by the time of submission but I'm happy to look into it if that is an option. You can watch [this](evaluation/tensorboard_meshes.webm) video that I recoreded from my screen where I show the reconstructions and the very bad generated samples. Note that I trained a couple of models with changes in different hyperparameters (latent space dimension, learning rate, diffusion steps etc) and all of them had a somewhat similar quality of results. I saved the parametrs that yields a slightly better result as default parameters in `train.py`. You can just run the code and should get similar results.


## Potential future improvements

I did not keep track of things one should do later but here are a few that comes to my mind:

1. Do some debugging to improve performance of the model, especially the generative component. Consider building a large model
2. Employ more sophisticated priors for the generative model (e.g. the prior used in LION from NVIDIA) to improve sample complexity and quality of outputs
3. Spent some time to figure out what the issue is for compiling the Chamfer Distance library (https://github.com/krrish94/chamferdist) to use it on GPU. I want to use this library instead of Pytorch3D's Chamfer Distance module but realized despite following the instructions from the author (I got in touch with the main author as well) for compiling, it throws errors when tensors are stored on the GPU. Will have to try it at some point later
4. Consider using a neural network-based EMD approximation method. The current method, despite running on the GPU using CUDA kernels, is highly computationally expensive. Also, check for potential normalization issues of the point clouds when computing EMD
5. Make a pipeline for simulating laser scanners (maybe consider converting point clouds to meshes with models like SAL or [this](https://huangjh-pub.github.io/publication/nksr/) work? and then manipulate them in Blender)
6. Debug the generative model. Consider sampling from empirical distribution
7. Use samples of the generative model or the laser scanner simulator to train a defect detection model
8. The outputs seem unnormalized. Figure out whether there is a normalization step that I'm missing in the training or during test time
9. Architectural changes: try an architecture based on [KANs](https://arxiv.org/abs/2404.19756) to make training more efficient

## Deliverables
 
I think I could accomplish the first part of deliverables but not sure how I should integrate this pipeline for metrology applications at the moment; I assume this is a goal for the future. Also, it would be easy to train a classifier for defect detection once the pipeline I worked on is ready and the generative model works well.