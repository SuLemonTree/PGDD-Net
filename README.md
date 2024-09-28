## 
<h2 align="center">SAM Is Not Always Perfect: When SAM Meets Industrial Defect Detection</h2>
<div align="center">
<p>PGDD-Net🚀 
是一种新颖的先验引导缺陷检测网络，该网络有效地利用SAM的知识来提高缺陷检测的鲁棒性和准确性。</p>
  <p>
    <a align="center" href="https://github.com/SuLemonTree/PGDD-Net" target="_blank">
      <img width="850" src="docs/image/yoloair.png"></a>
    <br><br>
  </p>


English | [简体中文](README_cn.md)
<p align="center">
<!--     <a href="https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a> -->
<!--     <a href="https://github.com/shaocong-qy/HFMRE/main/LICENSE">
        <img alt="license" src="https://github.com/shaocong-qy/HFMRE">
    </a> -->
    
    <a href="mailto: 10431220590@stu.qlu.edu.cn">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>



## Abstract
Since the introduction of distantly supervised relation extraction methods, numerous approaches have been developed, the most representative of which is multi-instance learning (MIL). To find reliable features that are most representative of multi-instance bags, aggregation strategies such as AVG (average), ONE (at least one), and ATT (sentence-level attention) are commonly used. These strategies tend to train third-party vectors to select sentence-level features, leaving it to the third party to decide/identify what is noise, ignoring the intrinsic associations that naturally exist from sentence to sentence. In this paper, we propose the concept of circular cosine similarity, which is used to explicitly show the intrinsic associations between sentences within a bag. We also consider the previous methods to be a crude denoising process as they are interrupted and do not have a continuous noise detection procedure. Following this consideration, we implement a relation extraction framework (HFMRE) that relies on the Huffman tree, where sentences are considered as leaf nodes and circular cosine similarity are considered as node weights. HFMRE can continuously and iteratively discriminate noise and aggregated features during the construction of the Huffman tree, eventually finding an excellent instance that is representative of a bag-level feature. The experiments demonstrate the remarkable effectiveness of our method, outperforming previously advanced baselines on the popular DSRE datasets.



<div align="center">
  <!-- <img src="https://github.com/qluinfo/HFMRE/blob/main/HFMRE_model.png" width=300 /> -->
  <img src="https://github.com/qluinfo/HFMRE/blob/main/HFMRE_model.png" width=800 >
</div>






This paper was published in ***EMNLP2023*** and is titled "[HFMRE: Constructing Huffman Tree in Bags to Find Excellent Instances for Distantly Supervised Relation Extraction](https://aclanthology.org/2023.findings-emnlp.854/)".




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
