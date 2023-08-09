# Model-Quantization



- [Large Language Model (LLM)](#large-language-model-llm)
- [Pre-trained Language Model (PLM)](#pre-trained-language-model-plm)
- [Low-bit Quantization](#low-bit-quantization)
- [Other](#other)

<!-- * [![Publish](https://img.shields.io/badge/<leaf tag>-<right tag>-<color>)]() 
[![Star](shields.io_url)](github_url) 
[paper title](paper url). 
some author. 
[[Paper]](paper url)
 [[Github]](github url)-->


## Large Language Model (LLM)

### 2023

* [![Publish](https://img.shields.io/badge/Conference-ICLR'22-blue)]() [![Star](https://img.shields.io/github/stars/IST-DASLab/gptq.svg?style=social&label=Star)](https://github.com/IST-DASLab/gptq) [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323). Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. [[Paper]](https://arxiv.org/abs/2210.17323) [[Github]](https://github.com/IST-DASLab/gptq)

* [![Star](https://img.shields.io/github/stars/qwopqwop200/GPTQ-for-LLaMa.svg?style=social&label=Star)](https://github.com/qwopqwop200/GPTQ-for-LLaMa) [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa): 4 bits quantization of LLaMA using GPTQ. [[Github]](https://github.com/qwopqwop200/GPTQ-for-LLaMa)

* [![Publish](https://img.shields.io/badge/Conference-ICML'23-blue)]() [![Star](https://img.shields.io/github/stars/mit-han-lab/smoothquant.svg?style=social&label=Star)](https://github.com/mit-han-lab/smoothquant) [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438). Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han. [[Paper]](https://arxiv.org/abs/2211.10438) [[Github]](https://github.com/mit-han-lab/smoothquant)

* [![Star](https://img.shields.io/github/stars/mit-han-lab/llm-awq.svg?style=social&label=Star)](https://github.com/mit-han-lab/llm-awq) [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978). Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Song Han. [[Paper]](https://github.com/mit-han-lab/llm-awq) [[Github]](https://github.com/mit-han-lab/llm-awq)

* [![Star](https://img.shields.io/github/stars/hahnyuan/RPTQ4LLM.svg?style=social&label=Star)](https://github.com/hahnyuan/RPTQ4LLM) [RPTQ: Reorder-based Post-training Quantization for Large Language Models](https://arxiv.org/abs/2304.01089). Zhihang Yuan and Lin Niu and Jiawei Liu and Wenyu Liu and Xinggang Wang and Yuzhang Shang and Guangyu Sun and Qiang Wu and Jiaxiang Wu and Bingzhe Wu. [[Paper]](https://arxiv.org/abs/2304.01089) [[Github]](https://github.com/hahnyuan/RPTQ4LLM)

* [![Star](https://img.shields.io/github/stars/artidoro/qlora.svg?style=social&label=Star)](https://github.com/artidoro/qlora) [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. [[Paper]](https://arxiv.org/abs/2305.14314) [[Github]](https://github.com/artidoro/qlora)

* [ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation](https://arxiv.org/abs/2303.08302). Zhewei Yao, Xiaoxia Wu, Cheng Li, Stephen Youn, Yuxiong He. [[Paper]](https://arxiv.org/abs/2303.08302)

* [![Star](https://img.shields.io/github/stars/SqueezeAILab/SqueezeLLM.svg?style=social&label=Star)](https://github.com/SqueezeAILab/SqueezeLLM) [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/pdf/2306.07629.pdf). Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, Kurt Keutzer. [[Paper]](https://arxiv.org/pdf/2306.07629.pdf) [[Github]](https://github.com/SqueezeAILab/SqueezeLLM)

* [Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling](https://arxiv.org/abs/2304.09145v1). Xiuying Wei , Yunchen Zhang, Yuhang Li, Xiangguo Zhang, Ruihao Gong, Jinyang Guo, Xianglong Liu. [[Paper]](https://arxiv.org/abs/2304.09145v1)

* [Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models](https://arxiv.org/abs/2305.12356). Yijia Zhang, Lingran Zhao, Shijie Cao, Wenqiang Wang, Ting Cao, Fan Yang, Mao Yang, Shanghang Zhang, Ningyi Xu. [[Paper]](https://arxiv.org/abs/2305.12356)

* [LLM-QAT: Data-Free Quantization Aware Training for Large Language Models](https://arxiv.org/abs/2305.17888). 
Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, Vikas Chandra. [[Paper]](https://arxiv.org/abs/2305.17888)

* [![Star](https://img.shields.io/github/stars/Vahe1994/SpQR.svg?style=social&label=Star)](https://github.com/Vahe1994/SpQR) [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078). Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, Dan Alistarh. [[Paper]](https://arxiv.org/abs/2306.03078)  [[Github]](https://github.com/Vahe1994/SpQR)

* [![Star](https://img.shields.io/github/stars/xvyaward/owq.svg?style=social&label=Star)](https://github.com/xvyaward/owq) [OWQ: Lessons learned from activation outliers for weight quantization in large language models](https://arxiv.org/abs/2306.02272). Changhun Lee, Jungyu Jin, Taesu Kim, Hyungjun Kim, Eunhyeok Park. [[Paper]](https://arxiv.org/abs/2306.02272) [[Github]](https://github.com/xvyaward/owq)

* [![Star](https://img.shields.io/github/stars/RUCAIBox/QuantizedEmpirical.svg?style=social&label=Star)](https://github.com/RUCAIBox/QuantizedEmpirical) [Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study](https://arxiv.org/abs/2307.08072). Peiyu Liu, Zikang Liu, Ze-Feng Gao, Dawei Gao, Wayne Xin Zhao, Yaliang Li, Bolin Ding, Ji-Rong Wen. [[Paper]](https://arxiv.org/abs/2307.08072) [[Github]](https://github.com/RUCAIBox/QuantizedEmpirical) 

* [ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats](https://arxiv.org/abs/2307.09782). Xiaoxia Wu, Zhewei Yao, Yuxiong He. [[Paper]](https://arxiv.org/abs/2307.09782)




## Pre-trained Language Model (PLM)

### 2023
<!-- from https://github.com/horseee/Awesome-Efficient-LLM/tree/main/efficient_plm -->
* [![Star](https://img.shields.io/github/stars/wimh966/outlier_suppression.svg?style=social&label=Star)](https://github.com/wimh966/outlier_suppression) [Outlier Suppression: Pushing the Limit of Low-bit Transformer](https://arxiv.org/abs/2209.13325). Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, Xianglong Liu. [[Paper]](https://arxiv.org/abs/2209.13325)[[Github]](https://github.com/wimh966/outlier_suppression)
* [![Publish](https://img.shields.io/badge/Conference-ACL'23-blue)]() [Self-Distilled Quantization: Achieving High Compression Rates in Transformer-Based Language Models](https://aclanthology.org/2023.acl-short.114/). James O’Neill, Sourav Dutta. [[Paper]](https://aclanthology.org/2023.acl-short.114/)
* [![Publish](https://img.shields.io/badge/Conference-ICML'23-blue)]() [Understanding Int4 Quantization for Language Models: Latency Speedup, Composability, and Failure Cases](https://openreview.net/forum?id=q1WGm3hItW). Xiaoxia Wu, Cheng Li, Reza Yazdani Aminabadi, Zhewei Yao, Yuxiong He. [[Paper]](https://openreview.net/forum?id=q1WGm3hItW)
* [![Publish](https://img.shields.io/badge/Conference-ACL'23%20Findings-blue)]() [PreQuant: A Task-agnostic Quantization Approach for Pre-trained Language Models](https://arxiv.org/abs/2306.00014). Zhuocheng Gong, Jiahao Liu, Qifan Wang, Yang Yang, Jingang Wang, Wei Wu, Yunsen Xian, Dongyan Zhao, Rui Yan. [[Paper]](https://arxiv.org/abs/2306.00014)
* [![Publish](https://img.shields.io/badge/Conference-ACL'23%20Findings-blue)]() [Boost Transformer-based Language Models with GPU-Friendly Sparsity and Quantization](https://aclanthology.org/2023.findings-acl.15.pdf). Chong Yu, Tao Chen, Zhongxue Gan. [[Paper]](https://aclanthology.org/2023.findings-acl.15.pdf)
* [Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing](https://arxiv.org/abs/2306.12929). Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort. [[Paper]](https://arxiv.org/abs/2306.12929)
<!-- from https://github.com/horseee/Awesome-Efficient-LLM/tree/main/efficient_plm -->

* [![Publish](https://img.shields.io/badge/Conference-CVPR'23-blue)]() [NoisyQuant: Noisy Bias-Enhanced Post-Training Activation Quantization for Vision Transformers](https://arxiv.org/abs/2211.16056). Yijiang Liu, Huanrui Yang, Zhen Dong, Kurt Keutzer, Li Du, Shanghang Zhang. [[Paper]](https://arxiv.org/abs/2211.16056)

* [![Publish](https://img.shields.io/badge/Conference-CVPR'23-blue)]() [Boost Vision Transformer with GPU-Friendly Sparsity and Quantization](https://arxiv.org/abs/2305.10727). Chong Yu, Tao Chen, Zhongxue Gan, Jiayuan Fan. [[Paper]](https://arxiv.org/abs/2305.10727)

* [Q-HyViT: Post-Training Quantization for Hybrid Vision Transformer with Bridge Block Reconstruction](https://arxiv.org/abs/2303.12557). Jemin Lee, Yongin Kwon, Jeman Park, Misun Yu, Hwanjun Song. [[Paper]](https://arxiv.org/abs/2303.12557)

* [LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models](https://arxiv.org/abs/2206.09557). Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, Dongsoo Lee. [[Paper]](https://arxiv.org/abs/2206.09557)


* [Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM Inference with Transferable Prompt](https://arxiv.org/abs/2305.11186). Zhaozhuo Xu, Zirui Liu, Beidi Chen, Yuxin Tang, Jue Wang, Kaixiong Zhou, Xia Hu, Anshumali Shrivastava. [[Paper]](https://arxiv.org/abs/2305.11186)


## Low-bit Quantization

### 2023


* [![Publish](https://img.shields.io/badge/Conference-WACV'23-blue)]() [Collaborative Multi-Teacher Knowledge Distillation for Learning Low Bit-width Deep Neural Networks](https://arxiv.org/abs/2210.16103). Cuong Pham, Tuan Hoang, Thanh-Toan Do. [[Paper]](https://arxiv.org/abs/2210.16103)

* [![Publish](https://img.shields.io/badge/Conference-CVPR'23-blue)]() [![Stars](https://img.shields.io/github/stars/SamsungLabs/Genie?style=social&label=star)](https://github.com/SamsungLabs/Genie) [Genie: Show Me the Data for Quantization](https://arxiv.org/abs/2212.04780). Yongkweon Jeon, Chungman Lee, Ho-young Kim. [[Paper]](https://arxiv.org/abs/2212.04780) [[Github]](https://github.com/SamsungLabs/Genie)




## Other

### 2023

* [![Publish](https://img.shields.io/badge/Conference-ICML'23-blue)]() [![Stars](https://img.shields.io/github/stars/htqin/BiBench?style=social&label=star)](https://github.com/htqin/BiBench) [BiBench: Benchmarking and Analyzing Network Binarization](https://arxiv.org/abs/2301.11233). Haotong Qin, Mingyuan Zhang, Yifu Ding, Aoyu Li, Zhongang Cai, Ziwei Liu, Fisher Yu, Xianglong Liu. [[Paper]](https://arxiv.org/abs/2301.11233) [[Github]](https://github.com/htqin/BiBench)

* [![Publish](https://img.shields.io/badge/Conference-CVPR'23-blue)]() [Toward Accurate Post-Training Quantization for Image Super Resolution](https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.pdf). Zhijun Tu, Jie Hu, Hanting Chen, Yunhe Wang. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.pdf)

* [![Publish](https://img.shields.io/badge/Conference-CVPR'23-blue)]() [One-Shot Model for Mixed-Precision Quantization](https://openaccess.thecvf.com/content/CVPR2023/papers/Koryakovskiy_One-Shot_Model_for_Mixed-Precision_Quantization_CVPR_2023_paper.pdf). Ivan Koryakovskiy. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Koryakovskiy_One-Shot_Model_for_Mixed-Precision_Quantization_CVPR_2023_paper.pdf)

* [![Publish](https://img.shields.io/badge/Conference-CVPR'23-blue)]() [![Stars](https://img.shields.io/github/stars/hfutqian/AdaDFQ?style=social&label=star)](https://github.com/hfutqian/AdaDFQ) [Adaptive Data-Free Quantization](https://arxiv.org/abs/2303.06869). Biao Qian, Yang Wang, Richang Hong, Meng Wang. [[Paper]](https://arxiv.org/abs/2303.06869) [[Github]](https://github.com/hfutqian/AdaDFQ)

* [![Publish](https://img.shields.io/badge/Conference-CVPR'23-blue)]() [![Stars](https://img.shields.io/github/stars/ECoLab-POSTECH/NIPQ?style=social&label=star)](https://github.com/ECoLab-POSTECH/NIPQ) [NIPQ: Noise proxy-based Integrated Pseudo-Quantization](https://arxiv.org/abs/2206.00820). Juncheol Shin, Junhyuk So, Sein Park, Seungyeop Kang, Sungjoo Yoo, Eunhyeok Park. [[Paper]](https://arxiv.org/abs/2206.00820) [[Github]](https://github.com/ECoLab-POSTECH/NIPQ)

* [![Publish](https://img.shields.io/badge/Conference-CVPR'23-blue)]() [Bit-shrinking: Limiting Instantaneous Sharpness for Improving Post-training Quantization](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Bit-Shrinking_Limiting_Instantaneous_Sharpness_for_Improving_Post-Training_Quantization_CVPR_2023_paper.pdf). Chen Lin. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Bit-Shrinking_Limiting_Instantaneous_Sharpness_for_Improving_Post-Training_Quantization_CVPR_2023_paper.pdf)

* [![Publish](https://img.shields.io/badge/Conference-CVPR'23-blue)]() [![Stars](https://img.shields.io/github/stars/bytedance/MRECG?style=social&label=star)](https://github.com/bytedance/MRECG) [Solving Oscillation Problem in Post-Training Quantization Through a Theoretical Perspective](https://arxiv.org/abs/2303.11906). Yuexiao Ma, Huixia Li, Xiawu Zheng, Xuefeng Xiao, Rui Wang, Shilei Wen, Xin Pan, Fei Chao, Rongrong Ji. [[Paper]](https://arxiv.org/abs/2303.11906) [[Github]](https://github.com/bytedance/MRECG)

* [![Publish](https://img.shields.io/badge/Conference-CVPR'23-blue)]() [![Stars](https://img.shields.io/github/stars/WooKyoungHan/ABCD?style=social&label=star)](https://github.com/WooKyoungHan/ABCD) [ABCD : Arbitrary Bitwise Coefficient for De-quantization](https://openaccess.thecvf.com/content/CVPR2023/papers/Han_ABCD_Arbitrary_Bitwise_Coefficient_for_De-Quantization_CVPR_2023_paper.pdf). Woo Kyoung Han. [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Han_ABCD_Arbitrary_Bitwise_Coefficient_for_De-Quantization_CVPR_2023_paper.pdf) [[Github]](https://github.com/WooKyoungHan/ABCD)

* [![Publish](https://img.shields.io/badge/Journal-TNNLS'23-blue)]() [![Stars](https://img.shields.io/github/stars/htqin/BiFSMNv2?style=social&label=star)](https://github.com/htqin/BiFSMNv2) [BiFSMNv2: Pushing Binary Neural Networks for Keyword Spotting to Real-Network Performance](https://arxiv.org/abs/2211.06987). Haotong Qin, Xudong Ma, Yifu Ding, Xiaoyang Li, Yang Zhang, Zejun Ma, Jiakai Wang, Jie Luo, Xianglong Liu. [[Paper]](https://arxiv.org/abs/2211.06987) [[Github]](https://github.com/htqin/BiFSMNv2)

* [![Publish](https://img.shields.io/badge/Journal-PR'23-blue)]() [Bayesian asymmetric quantized neural networks](https://www.sciencedirect.com/science/article/pii/S0031320323001632). Jen-Tzung Chien, Su-Ting Chang. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320323001632)

* [![Publish](https://img.shields.io/badge/Conference-MMM'23-blue)]() [Binary Neural Network for Video Action Recognition](https://link.springer.com/chapter/10.1007/978-3-031-27077-2_8). Hongfeng Han, Zhiwu Lu, Ji-Rong Wen. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-27077-2_8)

* [![Stars](https://img.shields.io/github/stars/YixuanSeanZhou/Quantized_Neural_Nets?style=social&label=star)](https://github.com/YixuanSeanZhou/Quantized_Neural_Nets) [Post-training Quantization for Neural Networks with Provable Guarantees](https://arxiv.org/abs/2201.11113). Jinjie Zhang, Yixuan Zhou, Rayan Saab. [[Paper]](https://arxiv.org/abs/2201.11113) [[Github]](https://github.com/YixuanSeanZhou/Quantized_Neural_Nets)

* [EBSR: Enhanced Binary Neural Network for Image Super-Resolution](https://arxiv.org/abs/2303.12270). Renjie Wei, Shuwen Zhang, Zechun Liu, Meng Li, Yuchen Fan, Runsheng Wang, Ru Huang. [[Paper]](https://arxiv.org/abs/2303.12270)

* [Binarizing Sparse Convolutional Networks for Efficient Point Cloud Analysis](https://arxiv.org/abs/2303.15493). Xiuwei Xu, Ziwei Wang, Jie Zhou, Jiwen Lu. [[Paper]](https://arxiv.org/abs/2303.15493)

* [Binary domain generalization for sparsifying binary neural networks](https://arxiv.org/abs/2306.13515). Riccardo Schiavone, Francesco Galati, Maria A. Zuluaga. [[Paper]](https://arxiv.org/abs/2306.13515)

























