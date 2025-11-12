# GFRIEND: Generative Few-shot Reward Inference through Efficient DPO

<a href='https://arxiv.org/abs/2506.08965'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

💻 This is the official implementation for our paper [*GFRIEND: Generative Few-shot Reward Inference through Efficient DPO*](https://arxiv.org/abs/2506.08965).

#### Authors
[Yiyang Zhao](),  [Zhiqi Shen](https://dr.ntu.edu.sg/entities/person/Shen-Zhiqi), [Xuejiao Zhao*](https://zxjwudi.github.io/xuejiaozhao/)

**Nanyang Technological University**

\* Corresponding author

---

## :fire: News
* **[2025.06.10]** We release the latest paper version on [arXiv](https://arxiv.org/abs/2506.08965).
* **[2025.06.09]** We have added more detailed information on the dataset, RewardBench, and data preprocessing. Have a try!
* **[2025.06.09]** We release the official implementation of **GFRIEND**.

## Overview
**GFRIEND** is a generative reward model for **RLHF** (Reinforcement Learning with Human Feedback) designed for scenarios with **limited human preference data**. Specifically, GFRIEND integrates a **preference refinement module** to produce diverse, high-quality preference data, mitigating data sparsity. Then, GFRIEND employs a **multi-level preference modeling strategy** rather than simple binary comparisons, using a perplexity-based scoring mechanism to quantify preference degrees and enable finer-grained reward modeling. Next, we **modify the Direct Preference Optimization (DPO)** loss by weighting sample pairs based on preference disparity, ensuring more representative data is emphasized during reward model training.

The core processes of GFRIEND include:
1. **SFT**: Supervised fine-tuning of the base model using a small amount of (question, chain-of-thought) data to enable it to generate high-quality thoughts/reasoning.
2. **Preference Refinement**: Sampling multiple times on data with preference labels to generate diverse CoT (chain-of-thought) and judgment results, and expanding and fine-grainedly distinguishing preference data based on perplexity scoring.
3. **M-DPO**: Weighted Direct Preference Optimization training on the above multi-level preference data.


<div align="left"> <img src="./fig/figxj.png" alt="diseases knowledge graph" width="420"> </div >
    <p><em><strong>Figure 1: The overall framework of GFRIEND.</strong> The steps for generating more preference data with a preference dataset that includes preference labels for a pair of answers to a question.</em></p>

<!-- ![The steps for Generating more preference datas with a preference dataset that includes preference labels for a pair of answers to a question.](./fig/fig9.png)   -->

<!-- *Figure 1: The steps for Generating more preference datas with a preference dataset that includes preference labels for a pair of answers to a question.* -->


## Dataset

The project primarily utilizes the following two types of datasets for training and evaluation as described in the paper:

1. **General Domain Dataset**: We selected the publicly available ["Skywork-Reward-Preference-80K-v0.2"]("Skywork-Reward-Preference-80K-v0.2") as the base preference data. For few-shot scenarios, we used a small number of high-quality samples (approximately 3,000) for experimentation and tested on public benchmarks such as

- [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench) A dataset for evaluating the capabilities of reward models, covering multiple categories including chat, reasoning, and safety, is designed to test the performance of reward models in complex and structured queries.
- [UltraFeedback](https://github.com/OpenBMB/UltraFeedback) A large-scale, fine-grained, and diverse preference dataset, containing prompts from various resources, and annotated by GPT-4 in four aspects: instruction following, authenticity, honesty, and usefulness.
- [PKU-SafeRLHF](https://github.com/PKU-Alignment/safe-rlhf) A human-annotated preference dataset, containing over 300,000 human-labeled comparison data points, covering preferences for usefulness and harmlessness, aimed at promoting research on the safe alignment of large language models.


2. **Medical Domain Dataset**: To verify the effectiveness of the method in specialized scenarios, the paper constructed a medical preference dataset simulating a low-resource environment based on the [iCliniq](https://www.icliniq.com/) dataset. The dataset consists of 3,500 entries, with 3,000 used for training and 500 for validation. The data is derived from anonymized segments of real clinical conversations and publicly available medical data. It has undergone deduplication, normalization, anonymization, and expert annotation to form a structured preference format of (question, answer_pos, answer_neg).

When reproducing or conducting research using the above datasets, please note the following points:

- The preprocessing and filtering methods for the general domain dataset are detailed in the paper and script comments. It is recommended to ensure that there is no overlap between the training and test sets before training.
- If you have other custom preference data (such as for question-answering or dialogue scenarios), you can also integrate it into the same process in the format of (question, answer_neg, answer_pos).

## Experimental Results

### Main Results

![1](./fig/fig2.png) 

*Table 1:Accuracy of models’ judges on the test sets of Ultra-Feedback, PKU-SafeRLHF, and Reward-Bench. BT-model, ArmoRM and GFRIEND are trained on 3000 samples of Skywork-Reward-Preference-80K-v0.2 based
on Llama3-8B-Instruct.*

![2](./fig/fig3.png) 

*Table 2: Evaluation of different language model bases using supervised fine-tuning (SFT), BTmodel, and the GFRIEND method on the three benchmarks: UltraFeedback, PKU-SafeRLHF and Reward-Bench. With the exception of SFT, the data used to train the model were all 3000 samples.*

![3](./fig/fig6.png) 

*Table 3(Left): Judgment accuracy of GFRIEND and other models on specific medical datasets. BT-model, ArmoRM and GFRIEND are trained on 3000 samples based on Llama3-8B-Instruct.*

*Table 4(Right): Judgment accuracy of GFRIEND and its variants. CoT-S indicates whether or not to use Preference Refinement. With the exception of SFT, the models are all trained on 3000 samples based on Llama3-8B-Instruct.*







## Project Structure
- `/data`: Scripts for data loading and processing
- `/models`: Core logic for models, trainers, etc.
- `/generate`: Functions related to generating diverse preference data, including CoT sampling and perplexity calculation
- `/utils`: General utility functions, such as log management
- `/run_sft.py`: Script for running SFT training
- `/run_preference_refinement.py`: Script for generating and scoring multi-level preference data
- `/run_m_dpo.py`: Script for executing multi-level preference weighted DPO training

## Environment Dependencies
- Python 3.8+
- PyTorch >= 1.13
- transformers >= 4.30
- [Optional] accelerate / deepspeed / flash-attention, and other optimization tools

Installation method:
```bash
pip install -r requirements.txt
```

# Quick Start

SFT: Prepare your (question, chain-of-thought) data, adjust the data path in run_sft.py, and run:
```bash
python run_sft.py
```

Preference Refinement: Prepare your (question, a–, a+) preference pairs, then run:
```bash
python run_preference_refinement.py
```

M-DPO: Training with the generated multi-level preference data using a multi-level preference weighted loss.

```bash
python run_m_dpo.py
```

## 📖 Citation

If you find our work useful, please consider citing our papers:

```bibtex

@misc{zhao2025gfriendgenerativefewshotreward,
      title={GFRIEND: Generative Few-shot Reward Inference through Efficient DPO}, 
      author={Yiyang Zhao and Huiyu Bai and Xuejiao Zhao},
      year={2025},
      eprint={2506.08965},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.08965}, 
}


