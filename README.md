# GDTS
Code for paper "GDTS: Goal-Guided Diffusion Model with Tree Sampling for Multi-Modal Pedestrian Trajectory Prediction"
By Ge Sun, Sheng Wang, Lei Zhu, Ming Liu and Jun Ma*.

**The code will be released soon.**

**Abstract:**
Accurate prediction of pedestrian trajectories is crucial for improving the safety of autonomous driving. However, this task is generally nontrivial due to the inherent stochasticity of human motion, which naturally requires the predictor to generate multi-modal prediction. Previous works leverage various generative methods, such as GAN and VAE, for pedestrian trajectory prediction. Nevertheless, these methods may suffer from mode collapse and relatively low-quality results. The denoising diffusion probabilistic model (DDPM) has recently been applied to trajectory prediction due to its simple training process and powerful reconstruction ability. However, current diffusion-based methods do not fully utilize input information and usually require many denoising iterations that lead to a long inference time or an additional network for initialization. To address these challenges and facilitate the use of diffusion models in multi-modal trajectory prediction, we propose GDTS, a novel Goal-Guided Diffusion Model with Tree Sampling for multi-modal trajectory prediction. Considering the "goal-driven" characteristics of human motion, GDTS leverages goal estimation to guide the generation of the diffusion network. A two-stage tree sampling algorithm is presented, which leverages common features to reduce the inference time and improve accuracy for multi-modal prediction. Experimental results demonstrate that our proposed framework achieves comparable state-of-the-art performance with real-time inference speed in public datasets.

