## BT4222 LLM-for-Recommendation System

This repo contains the code for NUS BT4222 module topic: 
LLM for Recommendation System

The code is modified from project `LLM4REC`

[Code Repo](https://github.com/anord-wang/LLM4REC)
[Paper](https://arxiv.org/abs/2402.09617)

We strongly encourage you to take a look at their papers, for a rough understanding of the purpose of each stages.

The system design is based on GPT2. There are three major scripts in this repo:
- `llm4rec_training.py` The model learns the pattern of user/item interaction, 
    item descriptions, and user reviews. This stage is called pre-training because 
    we are just asking the model to get familiar with the scenario.

- `llm4rec_finetuning.py` The model is required to make predictions in this stage,
    And the correctness will be judged, and tune the model to make better predictions.

- `llm4rec_evaluation.py` The evaluation step.

And More detailed LLM-Related codes are under `libs\`, take a read if you are interested.

The training & finetuning takes 1 * 20GB GPU (RTX4000Ada) and 30+hrs, thus, we provide checkpoints for each stages. 

If you only have laptop/small machine, you can run the evaluation, which only takes 1GB of memory, and 5 mins of running time.

If you have a GPU containing 20GB+ of memory, or if you can access to GPU resources like NUS SoC or other platforms, and you are interested in the training of LLM, you can run the training & finetune.

### Prepare running environment

#### Conda Environment

Make sure you have conda installed.

Run these steps to create a conda-env to run our code:
```
conda create -n bt4222llm4rec pip
conda activate bt4222llm4rec
pip install -r requirements.txt
```

#### Prepare GPT2 repo

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

The pre-processed dataset is prepared in advance. You can download from:

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

#### Prepare the tokenizer

We have prepared the pre-trained tokenizer for you, you can download from:

And put them under:
```
BT4222LLM4REC:
    - provided_tokenizer:
        - merges.txt
        - vocab_file.json
```

#### Prepare the checkpoints

We have prepared the pre-training and fine-tuning checkpoints, you can download from:

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

Then we are read to run the scripts

### Run Evaluation

Simply run:
```
python llm4rec_evaluation.py
```

You can see the Recall & NDCG metrics.

### Read the code

Just simply running the code cannot help you understand how the things work.

Although you don't need to run the training and finetuning, you are still required to read through them. 

Meanwhile, where the magic of LLM happens is how they can understand human language. You may also open the `libs` folder, check out the `data.py` for how the user-item interaction data is converted to prompt and feeded into model. And open the `dataset\luxury\item_texts\review.txt`
 for how the reviews are compiled into prompt. 

### Try Pre-training and Fine-tuning by yourself

If you happen to have resources, time and interest to run the pre-training and fine-tuning. You can find another dataset, follow the data pre-processing steps in `data_processing_scripts\`

Remember to change the dataset name, then:
```
python llm4rec_training.py 2>&1 | tee self_run_training.log
```
You may note this will run for long long time. You can use `screen` command to initialize a virtual terminal for this running. 

This step will produce `50GB` of intermediate data, mainly the checkpoints in the middle of training. 

You can find all the saved checkpoints from `checkpoints\self-running\pretrain`

Find a best model with smalled loss from the pretrain checkpoints, rename it to `xxxxxx_best.xxx`, as how the checkpoints are provided. 

Then you need to change the checkpoint loading part of `llm4rec_finetuning.py` to be loading from your own checkpoint. 

Next, run fine-tuning with:
```
python llm4rec_finetuning.py 2>&1 | tee self_run_finetuning.log
```
This will also run for long time, and produce about `40GB` of intermediate data. And you can find the models from `\checkpoints\self-running\finetuning\`