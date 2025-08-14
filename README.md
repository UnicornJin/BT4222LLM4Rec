## BT4222 LLM-for-Recommendation System

This repo contains the code for NUS BT4222 module topic: LLM for Recommendation System

The code is modified from project `LLM4REC` ([Code Repo](https://github.com/anord-wang/LLM4REC), [Paper](https://arxiv.org/abs/2402.09617) )

For a more detailed understanding of the topic, you can take a look at their paper. 

(Meanwhile, for the research direction of "LLM for Recomm Sys", there a awesome summary for the papers, you can take a look if you are interested in this direction: [LLM4Rec Awesome Papers](https://github.com/WLiK/LLM4Rec-Awesome-Papers) )

The system design is based on GPT2. For the training and fine-tuning stages, there are three major scripts in this repo:
- `llm4rec_training.py` The model learns the pattern of user/item interaction, 
    item descriptions, and user reviews. This stage is called pre-training because 
    we are just asking the model to get familiar with the scenario.

- `llm4rec_finetuning.py` The model is required to make predictions in this stage,
    And the correctness will be judged, and tune the model to make better predictions.

- `llm4rec_evaluation.py` The evaluation step.

Please note that the core idea of this project is not limited to training a model, but a more foundamental concept of how to apply LLMs to recommendation systems. 
**Please go through the notebook `BT4222-LLM4Rec.ipynb` for a detailed explanation.**

The training & finetuning takes 1 * 20GB GPU (RTX4000Ada) and 30+hrs, thus, we provide checkpoints for each stages. 
If you only have laptop/small machine, you can run the evaluation, which only takes 1GB of memory, and 5 mins of running time.

### Prepare running environment

#### Conda Environment
(The codes are tested on `Ubuntu 24.04`, but should also work on other Linux distributions.)

Make sure you have `conda` installed. 
(If not, you can instrall it from [Anaconda](https://www.anaconda.com/))

Run these steps to create a conda-env to run our code:
```
conda create -n bt4222llm4rec pip
conda activate bt4222llm4rec
pip install -r requirements.txt
```

#### Prepare GPT2 repo
To download the large files from HuggingFace, Git LFS is required.
Make sure you have Git LFS installed, if not, you can install with:
```
conda install -c conda-forge git-lfs
git lfs install
```

Then pull GPT2 repo & weights from HuggingFace (https://huggingface.co/openai-community/gpt2)

```
git clone https://huggingface.co/openai-community/gpt2
cd gpt2
git lfs pull
```

Note: This will download `11GB` of data into the `gpt2/` folder. Make sure you have stable network connection, and you may take a walk/break when waiting for downloading.

#### Prepare the dataset

The pre-processed dataset is prepared in advance. You can download from: [Google Drive](https://drive.google.com/drive/folders/1FW7K87cfu2fERMa-vLOqM_feotexUDzB?usp=sharing)

And then put them in current folder, e.g.: 
```
BT4222LLM4REC:
    - dataset:
        - luxury:
            - item_texts\
            - user_item_texts\
            - meta.pkl
            - ...
```

Meanwhile, we also provide the data pre-processing scripts, you can find them under `data_processing_scripts\`. If you are interested, you can take a read.

The data processing is to extract the graph-based relationship from the raw dataset, as the paper describes, this is necessary to adapt LLMs to recommendation tasks. Meanwhile, we also need to extract the textual information like item descriptions and user reviews to provide the model with rich context about users and items.

#### Prepare the tokenizer

As the paper describes, the tokenizer should be modified to handle the user/item IDs. 

We have prepared the pre-trained tokenizer for you, you can download from: [Google Drive](https://drive.google.com/drive/folders/1FW7K87cfu2fERMa-vLOqM_feotexUDzB?usp=sharing)

And put them under:
```
BT4222LLM4REC:
    - provided_tokenizer:
        - merges.txt
        - vocab_file.json
```

#### Prepare the checkpoints

We have prepared the pre-training and fine-tuning checkpoints, you can download from: [Google Drive](https://drive.google.com/drive/folders/1FW7K87cfu2fERMa-vLOqM_feotexUDzB?usp=sharing)

And put them under:
```
BT4222LLM4REC:
    - checkpoints:
        - finetune:
            - luxury:
                - collaborative-based\
                - content-based\
        - pretrain
            - luxury:
                - collaborative-based\
                - content-based\
```

Then we are ready to run the evaluation scripts. 
(Or you can also try to run the training and fine-tuning scripts by yourself, see below for details.)

### Run Evaluation

Simply run:
```
python llm4rec_evaluation.py
```

You can see the Recall & NDCG metrics.

### Read the code

**Just simply running the code cannot help you understand how the things work.**

Although you don't need to run the training and finetuning, you are still required to read through them. 
Please follow the notebook, and refer to the corresponding code files to understand how the project works. 

### Try Pre-training and Fine-tuning by yourself (optional)

If you happen to have resources, time and interest to run the pre-training and fine-tuning. You can download another dataset from [Amazon Reviews](https://amazon-reviews-2023.github.io/), follow the data pre-processing steps in `data_processing_scripts\`

Remember to change the dataset name in codes, then run:
```
python llm4rec_training.py 2>&1 | tee self_run_training.log
```
You may note this will run for long long time. (You can use `screen` command to maintain the running in background.)

This step may produce `50GB` of intermediate data on disk, mainly the checkpoints in the middle of training. 

You can find all the saved checkpoints from `checkpoints\self-running\pretrain`

Find a best model with smalled loss from the pretrain checkpoints, rename it to `xxxxxx_best.xxx`, as how the checkpoints are provided. 

Then you need to change the checkpoint loading part of `llm4rec_finetuning.py` to be loading from your own checkpoint, run fine-tuning with:
```
python llm4rec_finetuning.py 2>&1 | tee self_run_finetuning.log
```
This will also run for long time, and produce about `40GB` of intermediate data. And you can find the models from `\checkpoints\self-running\finetuning\`