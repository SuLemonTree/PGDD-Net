## 
<h2 align="center">SAM Is Not Always Perfect: When SAM Meets Industrial Defect Detection</h2>
<div align="center">
<p>PGDD-Net🚀 
是一种新颖的先验引导缺陷检测网络，该网络有效地利用SAM的知识来提高缺陷检测的鲁棒性和准确性。</p>
  <p>
    <a align="center" href="https://github.com/SuLemonTree/PGDD-Net" target="_blank">
    </a>
    <br><br>
  </p>


English | [简体中文](README_cn.md)
<p align="center">   
    <a href="mailto: 10431220602@stu.qlu.edu.cn">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>



## Abstract
<p>
Industrial surface detection defect is crucial for safeguarding product quality and enhancing production efficiency. Existing detectors suffer from inferior results in terms of robustness and performance due to the complexity of industrial scenarios and the diversity of industrial defects. The Segment Anything Model (SAM) is adept at visual segmentation, yet the gap in its industry-specific knowledge impedes its efficacy in detecting industrial product defects. Still, SAM's reliable foundational knowledge aids in the localization and semantic understanding of industrial defects. In this paper, we propose the novel Prior-Guided Defect Detection Network (PGDD-Net) that effectively harnesses SAM's knowledge to bolster the robustness and accuracy of defect detection. Specifically, we introduce Expert Prior Encoder (EPEncoder), which integrates defects features into SAM using a learnable adapter to fortify its prior knowledge. Secondly, the Multi-Receptive Field Mamba (MRF-Mamba) is developed for modeling the global and local features of defects. Additionally, we propose a Prior-Weighted Embedding (PWE) module, which adeptly integrates SAM's localization and semantic priors via a dynamic weight allocation scheme. Finally, we present the Dual-Domain Feature Aggregation (DDFA) module for denoising and edge enhancement of defect features. The experimental results demonstrate that the proposed method achieves state-of-the-art performance on 8 benchmark industrial defect datasets.</p>



<div align="center">
  <!-- <img src="https://github.com/qluinfo/HFMRE/blob/main/HFMRE_model.png" width=300 /> -->
  <img src="https://github.com/SuLemonTree/PGDD-Net/blob/main/images/PGDD-Net.png" width=800 >
</div>

## Result

<div align="center">
  <img src="https://github.com/SuLemonTree/PGDD-Net/blob/main/images/db.jpg" width=800 >
</div>

## Visualization Result
<div align="center">
  <img src="https://github.com/SuLemonTree/PGDD-Net/blob/main/images/qp.png" width=800 >
</div>

## Quick start


<summary>Install</summary>

```bash
Python 3.10
```

```bash
pip install -r requirements.txt
```

```bash
cd PGDD-Net/Mobile SAM
```

```bash
pip install -v -e .
```

<summary>Data</summary>

<p>Baidu Cloud Drive contains 8 industrial defect datasets, download link:
</p>
  <p>
    <br>https://pan.baidu.com/s/1uIsHc_DI_uoBdUNG60Smrg
   Extract code:230l<br>
  </p>

<summary>Training & Evaluation</summary>

```shell
# training 
python train.py
# Evaluation 
python val.py
```
## Acknowledgement
<details>
<summary>
<a href="https://github.com/ChaoningZhang/MobileSAM">MobileSAM</a> Faster Segment Anything: Towards Lightweight SAM for Mobile Applications[<b>bib</b>]
</summary>

```bibtex
@article{zhang2023faster,
  title={Faster segment anything: Towards lightweight sam for mobile applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```
</details>


