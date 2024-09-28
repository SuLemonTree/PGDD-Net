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
  <img src="https://github.com/qluinfo/HFMRE/blob/main/HFMRE_model.png" width=800 >
</div>

## Result

<div align="center">
  <img src="https://github.com/qluinfo/HFMRE/blob/main/image.png" width=800 >
</div>


## Quick start


<summary>Install</summary>

```bash
pip install -r requirements.txt
```






<summary>Data</summary>

- download_nyt10.sh
- download_nyt10m.sh
- download_wiki20m.sh






<summary>Training & Evaluation</summary>

```shell
# training 
train_nyt10d.sh
train_nyt10m.sh
train_wiki20m.sh
```








## Citation
If you use `HFMRE` in your work, please use the following BibTeX entries:
```
@inproceedings{li-etal-2023-hfmre,
    title = "{HFMRE}: Constructing {H}uffman Tree in Bags to Find Excellent Instances for Distantly Supervised Relation Extraction",
    author = "Li, Min  and
      Shao, Cong  and
      Li, Gang  and
      Zhou, Mingle",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.854",
    doi = "10.18653/v1/2023.findings-emnlp.854",
    pages = "12820--12832",
    abstract = "Since the introduction of distantly supervised relation extraction methods, numerous approaches have been developed, the most representative of which is multi-instance learning (MIL). To find reliable features that are most representative of multi-instance bags, aggregation strategies such as AVG (average), ONE (at least one), and ATT (sentence-level attention) are commonly used. These strategies tend to train third-party vectors to select sentence-level features, leaving it to the third party to decide/identify what is noise, ignoring the intrinsic associations that naturally exist from sentence to sentence. In this paper, we propose the concept of circular cosine similarity, which is used to explicitly show the intrinsic associations between sentences within a bag. We also consider the previous methods to be a crude denoising process as they are interrupted and do not have a continuous noise detection procedure. Following this consideration, we implement a relation extraction framework (HFMRE) that relies on the Huffman tree, where sentences are considered as leaf nodes and circular cosine similarity are considered as node weights. HFMRE can continuously and iteratively discriminate noise and aggregated features during the construction of the Huffman tree, eventually finding an excellent instance that is representative of a bag-level feature. The experiments demonstrate the remarkable effectiveness of our method, outperforming previously advanced baselines on the popular DSRE datasets.",
}
```
