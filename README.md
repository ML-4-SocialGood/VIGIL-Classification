# Baseline
* ERM
* 2018 - ICLR - CrossGrad - [Generalizing Across Domains via Cross-Gradient Training](https://openreview.net/forum?id=r1Dx7fbCW)
* 2020 - AAAI - DDAIG - [Deep Domain-Adversarial Image Generation for Domain Generalisation](https://arxiv.org/abs/2003.06054)
* 2021 - MM - NKD - [Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation](https://dl.acm.org/doi/abs/10.1145/3474085.3475434)
* 2021 - ICML - CLIP-ZS/LinearProbe - [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html)
* 2021 - ICLR - MixStyle - [Domain Generalization with MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp)
* 2022 - IJCAI - DomainMix - [Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913)
* 2022 - IJCV - CoOp - [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)
* 2022 - CVPR - CoCoOp - [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)
* 2022 - CVPR - EFDMix - [Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization](https://arxiv.org/abs/2203.07740)
* 2023 - ICCV - RISE - [A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_A_Sentence_Speaks_a_Thousand_Images_Domain_Generalization_through_Distilling_ICCV_2023_paper.html)
* 2024 - AAAI - SSPL -  [Symmetric Self-Paced Learning for Domain Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/29639)
* 2024 - IJCV - CLIP-Adapter - [CLIP-Adapter: Better Vision-Language Models with Feature Adapters](https://link.springer.com/article/10.1007/s11263-023-01891-x)
* 2025 - IJCAI - BOLD

# Datasets
* Digits
* PACS
* OfficeHome
* VLCS
* Terra Incognita
* NICO++
* DomainNet

# Sample Command

python train.py

                --gpu 1                                                 # Specify device
                --seed 995                                              # Random Seed
                --output-dir output/BOKD-RN50-NICO-autumn               # Output directory 
                --dataset NICO                                          # Specify dataset
                --source-domains dim grass outdoor rock water           # Source Domains
                --target-domains autumn                                 # Target Domain
                --model BOLD                                            # Model for training
                --model-config-file config/bokd.yaml                    # Config file for model
