# Baseline
* ERM
* 2018 - CrossGrad - [Generalizing Across Domains via Cross-Gradient Training](https://openreview.net/forum?id=r1Dx7fbCW)
* 2020 - DDAIG - [Deep Domain-Adversarial Image Generation for Domain Generalisation](https://arxiv.org/abs/2003.06054)
* 2021 - MixStyle - [Domain Generalization with MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp)
* 2021 - NKD - [Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation](https://dl.acm.org/doi/abs/10.1145/3474085.3475434)
* 2022 - DomainMix - [Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913)
* 2022 - EFDMix - [Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization](https://arxiv.org/abs/2203.07740)
* 2023 - RISE - [A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_A_Sentence_Speaks_a_Thousand_Images_Domain_Generalization_through_Distilling_ICCV_2023_paper.html)
* 2024 - SSPL -  [Symmetric Self-Paced Learning for Domain Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/29639)

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
