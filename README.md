# Denoised Non-local Neural Network for Semantic Segmentation

PyTorch code for DenoisedNL (TNNLS 2023).

Denoised Non-local Neural Network for Semantic Segmentation  
Qi Song, Jie Li, and Hao Guo   
TNNLS, 2023  
[[Paper](https://ieeexplore.ieee.org/abstract/document/10018899)]

Abstract: The non-local (NL) network has become a widely used technique for semantic segmentation, which computes an attention map to measure the relationships of each pixel pair. However, most of the current popular NL models tend to ignore the phenomenon that the calculated attention map appears to be very noisy, containing interclass and intraclass inconsistencies, which lowers the accuracy and reliability of the NL methods. In this article, we figuratively denote these inconsistencies as attention noises and explore the solutions to denoise them. Specifically, we inventively propose a denoised NL network, which consists of two primary modules, i.e., the global rectifying (GR) block and the local retention (LR) block, to eliminate the interclass and intraclass noises, respectively. First, GR adopts the class-level predictions to capture a binary map to distinguish whether the selected two pixels belong to the same category. Second, LR captures the ignored local dependencies and further uses them to rectify the unwanted hollows in the attention map. The experimental results on two challenging semantic segmentation datasets demonstrate the superior performance of our model. Without any external training data, our proposed denoised NL can achieve the state-of-the-art performance of 83.5% and 46.69% mean of classwise intersection over union (mIoU) on Cityscapes and ADE20K, respectively. 

## Requirements:
```
pytorch==1.6.0
```

## Citation
If you found our method useful in your research, please consider citing

```
@article{song2023denoised,
  title={Denoised non-local neural network for semantic segmentation},
  author={Song, Qi and Li, Jie and Guo, Hao and Huang, Rui},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```
