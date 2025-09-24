import re
import os
import sys
import pickle
import fsspec
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import GPT2Config

sys.path.append("libs")
from libs.modeling_gpt2 import GPT2ModelWithBC
from libs.tokenizer import TokenizerWithUserItemIDTokensBatch
from libs.data import CollaborativeGPTGeneratorBatch
from libs.data import UserItemContentGPTDatasetBatch
from libs.model import GPT4RecommendationBaseModel
from libs.model import CollaborativeGPTwithItemLMHeadBatch
from libs.model import ContentGPTForUserItemWithLMHeadBatch

# ------------------------------------------------
# This script is to show the training process of 
# BT4222 LLM for Recommendation Example Code
# 
# The script is based on project 
# 'LLM4REC' https://github.com/anord-wang/LLM4REC
#
# Edition: 2025 Aug by Jin Yuze
# ------------------------------------------------

def save_local(remote_path, local_path, remote_mode, local_mode):
    '''
        Save the remote file in remote_path
        to the local_path...
    '''
    with fsspec.open(remote_path, remote_mode) as f:
        content = f.read()
    with fsspec.open(local_path, local_mode) as f:
        f.write(content)

def save_remote(local_path, remote_path, local_mode, remote_mode):
    '''
        Save the local file in local_path
        to the remote_path...
    '''
    with fsspec.open(local_path, local_mode) as f:
        content = f.read()
    with fsspec.open(remote_path, remote_mode) as f:
        f.write(content)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Set Up the environment, data paths, and configurations +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Environment Settings for CUDA's GPU
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# The Amazon Dataset we are using
dataset = 'luxury'
lambda_V = 1.0 # This is a hyperparameter of LLM4Rec

# Dataset Related file paths
# These are the paths to our pre-processed dataset
dataset_path = './dataset'
data_root = os.path.join(dataset_path, dataset)
meta_path = os.path.join(data_root, "meta.pkl")
filepath_list = [os.path.join(data_root, "user_item_texts", "review.pkl"),
                     os.path.join(data_root, "item_texts", "title.pkl"),
                     os.path.join(data_root, "item_texts", "brand.pkl"),
                     os.path.join(data_root, "item_texts", "categories.pkl"),
                     os.path.join(data_root, "item_texts", "description.pkl"),
                     os.path.join(data_root, "item_texts", "brand_extension.pkl"),
                     os.path.join(data_root, "item_texts", "categories_extension.pkl")
                     ]
review_path = os.path.join(data_root, "user_item_texts", "review.pkl")
train_matrix_path = os.path.join(data_root, "train_matrix.npz")

# The checkpoint main path
pre_train_checkpoint = os.path.join('./checkpoints', 'pretrain', dataset)

# The author's provided tokenizer
provided_tokenizer_path = './provided_tokenizer'
provided_vocab_file = os.path.join(provided_tokenizer_path, "vocab_file.json")
provided_merges_file = os.path.join(provided_tokenizer_path, "merges.txt")
local_vocab_file = os.path.join(pre_train_checkpoint, "vocab_file.json")
local_merges_file = os.path.join(pre_train_checkpoint, "merges.txt")

# GPT2 Pretrained Checkpoint 
gpt2_pretrained_path = './gpt2'
official_pretrained_weights_path = os.path.join(gpt2_pretrained_path, "pytorch_model.bin")
bt4222_pretrained_weights_path = os.path.join(pre_train_checkpoint, "gpt2", "pytorch_model.bin")

# The paths to save checkpoints, if you run by yourself
self_running_model_save_dir = os.path.join(pre_train_checkpoint, "self-running", "pretrain")
content_based_model_save_path = os.path.join(self_running_model_save_dir, dataset, "content-based")
collaborative_model_save_path = os.path.join(self_running_model_save_dir, dataset, "collaborative-based")
if not os.path.exists(content_based_model_save_path):
    os.makedirs(content_based_model_save_path, exist_ok=True)
if not os.path.exists(collaborative_model_save_path):
    os.makedirs(collaborative_model_save_path, exist_ok=True)

# configurations for the GPT2 model
_config = {
    "activation_function": "gelu_new",
    "architectures": [
        "GPT2LMHeadModel"
    ],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {
        "text-generation": {
            "do_sample": True,
            "max_length": 50
        }
    },
    "vocab_size": 50257
}
# +++++++++++++++++++++++++++++++++++++++


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# The training progress
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main():
    
    # Define the accelerator
    # the accelerator is used to handle the distributed training
    # but for this example, we actually just using single GPU.
    accelerator = Accelerator()
    # accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    accelerator.print("-----Current Settings-----")
    accelerator.print(f"Using dataset: {dataset}")
    accelerator.print(f"lambda_V: {lambda_V}")
    
    # Get the basic information of the dataset
    accelerator.print("-----Dataset Info-----")
    with fsspec.open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
    num_users = meta_data["num_users"]
    num_items = meta_data["num_items"]
    accelerator.print(f"Number of Users: {num_users}")
    accelerator.print(f"Number of Items: {num_items}")

    # Load the tokenizer with user/item tokens
    accelerator.print("-----Loading the Tokenizer-----")
    accelerator.print(f"Loading pretrained tokenizer from {provided_tokenizer_path}")
    if accelerator.is_main_process:
        save_local(provided_vocab_file, local_vocab_file, "r", "w")
        save_local(provided_merges_file, local_merges_file, "r", "w")
    accelerator.wait_for_everyone()

    # Define the tokenizer
    tokenizer = TokenizerWithUserItemIDTokensBatch(local_vocab_file, local_merges_file, num_users, num_items)

    # Load graph-based data for Interaction part, you can see the data is defined as a interation matrix
    accelerator.print("-----Loading Graph Data (Interaction)-----")
    mapping_graph_bc = torch.zeros((num_users + num_items, num_users + num_items), dtype=torch.float32)
    review_data_gen = UserItemContentGPTDatasetBatch(tokenizer, filepath_list, mapping_graph_bc)

    # Load graph-based data for collaborative part, which is the: user/item interaction data
    accelerator.print("-----Loading Interaction Data (Collaborative-based Data)-----")
    train_matrix = load_npz(train_matrix_path)
    collaborative_data_gen = CollaborativeGPTGeneratorBatch(tokenizer, train_matrix, mapping_graph_bc)

    # The config of the original GPT model needs a bit edition
    # The changing is simple, just add two attributes: num_users, num_items
    accelerator.print("-----Begin Setting Up the Config-----")
    config = GPT2Config(**_config)
    config.num_users = num_users
    config.num_items = num_items

    # Load the 'pre-trained' GPT2 model
    accelerator.print("-----Begin Instantiating the Pretrained GPT Model-----")
    gpt2model = GPT2ModelWithBC(config)
    gpt2model.load_state_dict(torch.load(official_pretrained_weights_path, weights_only=False), strict=False)

    # Instantiate the GPT for recommendation content-based model
    accelerator.print("-----Begin Instantiating the Content-based GPT Model-----")
    content_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    content_model = ContentGPTForUserItemWithLMHeadBatch(config, content_base_model)

    # [Optional]
    # Freeze the parameters of the pretrained GPT2 for content model
    # for name, param in content_model.named_parameters():
    #     # we allow only user/item token embeddings to be trained
    #     if ('user_embeddings' not in name) and \
    #             ('item_embeddings' not in name):
    #         param.requires_grad = False

    accelerator.print("-----Showing the Trainable Parameters-----")
    for name, param in content_model.named_parameters():
        if param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))

    accelerator.print("-----Showing the Non-trainable Parameters-----")
    for name, param in content_model.named_parameters():
        if not param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))

    # Instantiate the GPT for recommendation collaborative-based model
    accelerator.print("-----Begin Instantiating the Collaborative-based GPT Model-----")
    collaborative_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    collaborative_model = CollaborativeGPTwithItemLMHeadBatch(config, collaborative_base_model)

    # [Optional]
    # Freeze the parameters of the pretrained GPT2 for collaborative model
    # for name, param in collaborative_model.named_parameters():
    #     # we allow only user/item token embeddings to be trained
    #     if ('user_embeddings' not in name) and \
    #             ('item_embeddings' not in name):
    #         param.requires_grad = False

    accelerator.print("-----Showing the Trainable Parameters-----")
    for name, param in collaborative_model.named_parameters():
        if param.requires_grad:
            print("{} : {}".format(name, param.shape))

    accelerator.print("-----Showing the Non-Trainable Parameters-----")
    for name, param in collaborative_model.named_parameters():
        if not param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))

    # Set up the training details
    accelerator.print("-----Setting Up the Training Details-----")

    learning_rate = 1e-3
    batch_size_content_based = 4
    batch_size_collaborative_based = 32
    num_pretrained_epochs_for_content_based = 3
    num_pretrained_epochs_for_collaborative_based = 100

    accelerator.print("learning_rate: ", learning_rate)
    accelerator.print("batch_size_content_based: ", batch_size_content_based)
    accelerator.print("batch_size_collaborative_based: ", batch_size_collaborative_based)
    accelerator.print("num_pretrained_epochs_for_content_based: ", num_pretrained_epochs_for_content_based)
    accelerator.print("num_pretrained_epochs_for_collaborative_based: ", num_pretrained_epochs_for_collaborative_based)

    # Create the data loader
    num_workers=1 # We are using single GPU
    review_data_loader = DataLoader(review_data_gen,
                                    batch_size=batch_size_content_based,
                                    shuffle=True,
                                    collate_fn=review_data_gen.collate_fn,
                                    num_workers=num_workers)
    collaborative_data_loader = DataLoader(collaborative_data_gen,
                                           batch_size=batch_size_collaborative_based,
                                           collate_fn=collaborative_data_gen.collate_fn,
                                           num_workers=num_workers)
    accelerator.print("-----DataLoader Created-----")

    # Set the model to the training mode
    content_model.train()
    content_model.to(device)
    collaborative_model.train()
    collaborative_model.to(device)

    # Obtain the optimizer
    review_optimizer = optim.Adam(content_model.parameters(), lr=learning_rate)
    collaborative_optimizer = optim.Adam(collaborative_model.parameters(), lr=learning_rate)

    # model, optimizer and data loader with accelerator
    content_model, review_optimizer, review_data_loader = \
        accelerator.prepare(content_model, review_optimizer, review_data_loader)

    # model, optimizer and data loader with accelerator
    collaborative_model, collaborative_optimizer, collaborative_data_loader = \
        accelerator.prepare(collaborative_model, collaborative_optimizer, collaborative_data_loader)

    # Initialize best_loss with infinity
    review_best_loss = float('inf')
    collaborative_best_loss = float('inf')

    # The places to save the pre-training checkpoints
    accelerator.print(f"Content-based model: Weights will be saved to {content_based_model_save_path}")
    accelerator.print(f"Collaborative-based model: Weights will be saved to {collaborative_model_save_path}")

    # Finished doing the setup

    # ----------------------------------------------------------------
    # The pretraining loop for the content-based GPT begins from here
    # ----------------------------------------------------------------
    accelerator.print("-----Begin Content-based GPT Pretraining-----")
    
    for epoch in range(num_pretrained_epochs_for_content_based):
        review_total_loss = 0

        print(f'Epoch {epoch + 1}/{num_pretrained_epochs_for_content_based}')
        progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1}",
                            disable=not accelerator.is_local_main_process)

        for input_ids_prompt, input_ids_main, attention_mask, graph_bc_prompt, graph_bc_combined in progress_bar:
            review_optimizer.zero_grad()

            # Obtain the data, send to GPU
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)
            graph_bc_prompt = graph_bc_prompt.to(device)
            graph_bc_combined = graph_bc_combined.to(device)

            # Forward pass
            outputs = content_model(input_ids_prompt,
                                    input_ids_main,
                                    mapping_graph_bc_prompt=graph_bc_prompt,
                                    mapping_graph_bc_combined=graph_bc_combined,
                                    labels_main=input_ids_main,
                                    attention_mask=attention_mask)
            review_loss = outputs[0]

            # Backward pass and optimization
            accelerator.backward(review_loss)
            review_optimizer.step()

            review_total_loss += review_loss.item()
            progress_bar.set_postfix({"Content-based model, Loss": review_loss.item()})

        thread_review_average_loss = torch.tensor([review_total_loss / len(review_data_loader)]).to(device)
        gathered_review_average_loss = accelerator.gather(thread_review_average_loss)
        review_average_loss = torch.mean(gathered_review_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Content-based model, Average Loss: {review_average_loss:.4f}")

        # Check if the current loss is better than the best_loss
        # if so, save the best epoch
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss
            # Save the user & item embeddings
            user_emb_path = os.path.join(content_based_model_save_path, f"user_embeddings_{review_average_loss}_{epoch}.pt")
            item_emb_path = os.path.join(content_based_model_save_path, f"item_embeddings_{review_average_loss}_{epoch}.pt")
            gpt_save_path = os.path.join(content_based_model_save_path, f"content_based_gpt2_{review_average_loss}_{epoch}.bin")
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(content_model).base_model.user_embeddings.state_dict(), user_emb_path)
                torch.save(accelerator.unwrap_model(content_model).base_model.item_embeddings.state_dict(), item_emb_path)
                torch.save(accelerator.unwrap_model(content_model).base_model.gpt2model.state_dict(), gpt_save_path)
    
    accelerator.print("-----End Content GPT Pretraining Loop-----")
    
    # ----------------------------------------------------------------
    # The pretraining loop for the collaborative-based GPT begins from here    
    # ----------------------------------------------------------------
    accelerator.print("-----Begin Collaborative-based GPT Pretraining-----")

    for epoch in range(num_pretrained_epochs_for_collaborative_based):
        collaborative_total_loss = 0
        regularize_total_loss = 0

        accelerator.print(f'Epoch {epoch + 1}/{num_pretrained_epochs_for_collaborative_based}')
        progress_bar = tqdm(collaborative_data_loader, desc=f"Epoch {epoch + 1}",
                            disable=not accelerator.is_local_main_process, ncols=100)
        
        for input_ids_prompt, input_ids_main, attention_mask, graph_bc_prompt, graph_bc_combined in progress_bar:
            collaborative_optimizer.zero_grad()

            # Obtain the data, send to GPU
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)
            graph_bc_prompt = graph_bc_prompt.to(device)
            graph_bc_combined = graph_bc_combined.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                content_embeds = torch.cat(
                    (accelerator.unwrap_model(content_model).base_model.embed(input_ids_prompt),
                    accelerator.unwrap_model(content_model).base_model.embed(input_ids_main)),
                    axis=1
                ).to(device)

            # Forward pass
            outputs = collaborative_model(input_ids_prompt,
                                        input_ids_main,
                                        mapping_graph_bc_prompt=graph_bc_prompt,
                                        mapping_graph_bc_combined=graph_bc_combined,
                                        labels_main=input_ids_main,
                                        attention_mask=attention_mask,
                                        regularize=True,
                                        lambda_V=lambda_V,
                                        content_embeds=content_embeds)
            collaborative_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            accelerator.backward(collaborative_loss)
            collaborative_optimizer.step()

            collaborative_total_loss += collaborative_loss.item()
            regularize_total_loss += regularize_loss.item()

            progress_bar.set_postfix(
                {"Collaborative-based model, collaborative_loss": collaborative_loss.item(), 
                "Collaborative-based model, regularize_loss": regularize_loss.item()
                })

        # Gather the collaborative LM loss from different device
        thread_collaborative_average_loss = torch.tensor([collaborative_total_loss / len(collaborative_data_loader)]).to(device)
        gathered_collaborative_average_loss = accelerator.gather(thread_collaborative_average_loss)
        collaborative_average_loss = torch.mean(gathered_collaborative_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Content-based model, Average Collaborative Loss: {collaborative_average_loss:.4f}")

        # Gather the regularize loss from difference device
        thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(collaborative_data_loader)]).to(device)
        gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
        regularize_average_loss = torch.mean(gathered_regularize_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Content-based model, Average Regularize Loss: {regularize_average_loss:.4f}")

        # Check if the current loss is better than the best_loss
        if collaborative_average_loss < collaborative_best_loss:
            collaborative_best_loss = collaborative_average_loss
            # Save user & item embeddings
            user_emb_path = os.path.join(collaborative_model_save_path, f"user_embeddings_{collaborative_average_loss}_{epoch}.pt")
            item_emb_path = os.path.join(collaborative_model_save_path, f"item_embeddings_{collaborative_average_loss}_{epoch}.pt")
            gpt_save_path = os.path.join(collaborative_model_save_path, f"collaborative_based_gpt2_{collaborative_average_loss}_{epoch}.bin")
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(collaborative_model).base_model.user_embeddings.state_dict(), user_emb_path)
                torch.save(accelerator.unwrap_model(collaborative_model).base_model.item_embeddings.state_dict(), item_emb_path)
                torch.save(accelerator.unwrap_model(content_model).base_model.gpt2model.state_dict(), gpt_save_path)

        # According to the paper LLM4Rec section 3.3 
        # (This is mentioned in the fisrt version, the authors seems removed this from the latest version)
        # Previously the 3 epochs of content-based training is called "L-step"
        # And here, every 50 epochs of collaborative-based model training, 
        # there will be a "C-step" training for content-based model
        if (epoch + 1) % 50 == 0:
            accelerator.print("-----C-Step for Content-based GPT-----")
            review_total_loss = 0
            regularize_total_loss = 0

            progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1}",
                                disable=not accelerator.is_local_main_process, ncols=100)

            for input_ids_prompt, input_ids_main, attention_mask, graph_bc_prompt, graph_bc_combined in progress_bar:
                review_optimizer.zero_grad()

                input_ids_prompt = input_ids_prompt.to(device)
                input_ids_main = input_ids_main.to(device)
                attention_mask = attention_mask.to(device)
                graph_bc_prompt = graph_bc_prompt.to(device)
                graph_bc_combined = graph_bc_combined.to(device)

                accelerator.wait_for_everyone()
                with torch.no_grad():
                    collaborative_embeds = \
                        accelerator.unwrap_model(collaborative_model).base_model.embed(input_ids_prompt).to(device)

                # Forward pass
                outputs = content_model(input_ids_prompt,
                                        input_ids_main,
                                        mapping_graph_bc_prompt=graph_bc_prompt,
                                        mapping_graph_bc_combined=graph_bc_combined,
                                        labels_main=input_ids_main,
                                        attention_mask=attention_mask,
                                        regularize=True,
                                        lambda_V=lambda_V,
                                        collaborative_embeds=collaborative_embeds)
                review_loss = outputs[0]
                regularize_loss = outputs[1]

                # Backward pass and optimization
                accelerator.backward(review_loss)
                review_optimizer.step()

                review_total_loss += review_loss.item()
                regularize_total_loss += regularize_loss.item()

            # Gather the content LM loss from different device
            thread_review_average_loss = torch.tensor([review_total_loss / len(review_data_loader)]).to(device)
            gathered_review_average_loss = accelerator.gather(thread_review_average_loss)
            review_average_loss = torch.mean(gathered_review_average_loss)
            accelerator.print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")

            # Gather the regularize loss from different device
            thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(review_data_loader)]).to(device)
            gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
            regularize_average_loss = torch.mean(gathered_regularize_average_loss)
            accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

            # Check if the current loss is better than the best_loss
            accelerator.wait_for_everyone()
            if review_average_loss < review_best_loss:
                review_best_loss = review_average_loss
                # Save the user & item embeddings
                user_emb_path = os.path.join(content_based_model_save_path, f"user_embeddings_{review_average_loss}_{epoch}.pt")
                item_emb_path = os.path.join(content_based_model_save_path, f"item_embeddings_{review_average_loss}_{epoch}.pt")
                gpt_save_path = os.path.join(content_based_model_save_path, f"content_based_gpt2_{review_average_loss}_{epoch}.bin")
                if accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(content_model).base_model.user_embeddings.state_dict(), user_emb_path)
                    torch.save(accelerator.unwrap_model(content_model).base_model.item_embeddings.state_dict(), item_emb_path)
                    torch.save(accelerator.unwrap_model(content_model).base_model.gpt2model.state_dict(), gpt_save_path)

if __name__ == "__main__":
    main()
