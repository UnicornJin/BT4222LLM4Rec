import re
import os
import sys
import pickle
import fsspec
import random
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
from libs.data import UserItemContentGPTDatasetBatch
from libs.data import RecommendationGPTTrainGeneratorBatch
from libs.data import RecommendationGPTTestGeneratorBatch
from libs.model import GPT4RecommendationBaseModel
from libs.model import ContentGPTForUserItemWithLMHeadBatch
from libs.model import CollaborativeGPTwithItemRecommendHead
from libs.util import Recall_at_k, NDCG_at_k

# ------------------------------------------------
# This script is to show the fine-tuning process of 
# BT4222 LLM for Recommendation Example Code
# 
# The script is based on project 
# 'LLM4REC' https://github.com/anord-wang/LLM4REC
#
# Edition: 2025.05.01 by Jin Yuze
# ------------------------------------------------

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Set Up the environment, data paths, and configurations +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Environment Settings for CUDA's GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# The Amazon Dataset we are using
dataset = 'luxury'
lambda_V = 1.0

# Dataset Related file paths
# These are the paths to our pre-processed dataset
dataset_path = './dataset'
data_root = os.path.join(dataset_path, dataset)
meta_path = os.path.join(data_root, "meta.pkl")
mapping_graph_bc_path = os.path.join(data_root, "interaction_matrix.npz")
review_path = os.path.join(data_root, "user_item_texts", "review.pkl")
filepath_list = [os.path.join(data_root, "user_item_texts", "review.pkl"),
                    os.path.join(data_root, "item_texts", "title.pkl"),
                    os.path.join(data_root, "item_texts", "brand.pkl"),
                    os.path.join(data_root, "item_texts", "categories.pkl"),
                    os.path.join(data_root, "item_texts", "description.pkl"),
                    os.path.join(data_root, "item_texts", "brand_extension.pkl"),
                    os.path.join(data_root, "item_texts", "categories_extension.pkl"),
                    ]
train_mat_path = os.path.join(data_root, "train_matrix.npz")
val_mat_path = os.path.join(data_root, "val_matrix.npz")

# The checkpoint main path
pre_train_checkpoint = os.path.join('./checkpoints', 'pretrain', dataset)
# The paths to the pre-trained weights we provided
bt4222_content_based_gpt2_pretrained_weights_path = os.path.join(pre_train_checkpoint, "content-based", "content_based_gpt2_best.bin")
bt4222_content_based_gpt2_pretrained_user_emb_path = os.path.join(pre_train_checkpoint, "content-based", "user_embeddings_best.pt")
bt4222_content_based_gpt2_pretrained_item_emb_path = os.path.join(pre_train_checkpoint, "content-based", "item_embeddings_best.pt")
bt4222_collaborative_based_gpt2_pretrained_weights_path = os.path.join(pre_train_checkpoint, "collaborative-based", "collaborative_based_gpt2_best.bin")
bt4222_collaborative_based_gpt2_pretrained_user_emb_path = os.path.join(pre_train_checkpoint, "collaborative-based", "user_embeddings_best.pt")
bt4222_collaborative_based_gpt2_pretrained_item_emb_path = os.path.join(pre_train_checkpoint, "collaborative-based", "item_embeddings_best.pt")

# The fine-tune checkpoint main path
fine_tune_checkpoint = os.path.join('./checkpoints', 'finetune', dataset)

# The author's provided tokenizer
provided_tokenizer_path = './provided_tokenizer'
provided_vocab_file = os.path.join(provided_tokenizer_path, "vocab_file.json")
provided_merges_file = os.path.join(provided_tokenizer_path, "merges.txt")

# The paths to save checkpoints, if you run by yourself
self_running_model_save_dir = os.path.join(pre_train_checkpoint, "self-running", "finetune")
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
# The fine-tune progress
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

    # Obtain the tokenizer with user/item tokens
    accelerator.print("-----Loading the Tokenizer-----")
    accelerator.print(f"Loading pretrained tokenizer from {provided_tokenizer_path}...")
    tokenizer = TokenizerWithUserItemIDTokensBatch(provided_vocab_file, provided_merges_file, num_users, num_items)
    
    # Load data content-based data
    accelerator.print("-----Loading Review Data (Content-based Data)-----")
    mapping_graph_bc = load_npz(mapping_graph_bc_path)
    content_data_gen = UserItemContentGPTDatasetBatch(tokenizer, filepath_list, mapping_graph_bc)

    # Load the training&validation data generator
    accelerator.print("-----Load Train&Validate Collaborative-based Data-----")
    train_mat = load_npz(train_mat_path)
    val_mat = load_npz(val_mat_path)
    collaborative_train_data_gen = RecommendationGPTTrainGeneratorBatch(tokenizer, train_mat, mapping_graph_bc)
    collaborative_val_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, val_mat, mapping_graph_bc)
    
    # The config of the original GPT model needs a bit edition
    # The changing is simple, just add two attributes: num_users, num_items
    accelerator.print("-----Begin Setting Up the Config-----")
    config = GPT2Config(**_config)
    config.num_users = num_users
    config.num_items = num_items

    # Instantiate the GPT for recommendation content-based model
    accelerator.print("-----Begin Instantiating the Content-based GPT Model-----")
    gpt2model = GPT2ModelWithBC(config)
    gpt2model.load_state_dict(torch.load(bt4222_content_based_gpt2_pretrained_weights_path, map_location=device), strict=False)
    base_model = GPT4RecommendationBaseModel(config, gpt2model)
    base_model.user_embeddings.load_state_dict(torch.load(bt4222_content_based_gpt2_pretrained_user_emb_path, map_location=device))
    base_model.item_embeddings.load_state_dict(torch.load(bt4222_content_based_gpt2_pretrained_item_emb_path, map_location=device))
    content_model = ContentGPTForUserItemWithLMHeadBatch(config, base_model)    

    # Instantiate the GPT for recommendation collaborative-based model (target fine-tuned model)
    accelerator.print("-----Begin Instantiating the Collaborative-based, targeted Fine-tuned GPT Model-----")
    gpt2model = GPT2ModelWithBC(config)
    gpt2model.load_state_dict(torch.load(bt4222_collaborative_based_gpt2_pretrained_weights_path, map_location=device), strict=False)
    base_model = GPT4RecommendationBaseModel(config, gpt2model)
    base_model.user_embeddings.load_state_dict(torch.load(bt4222_collaborative_based_gpt2_pretrained_user_emb_path, map_location=device))
    base_model.item_embeddings.load_state_dict(torch.load(bt4222_collaborative_based_gpt2_pretrained_item_emb_path, map_location=device))
    collaborate_model = CollaborativeGPTwithItemRecommendHead(config, base_model)

    # [Optional]
    # Freeze the parameters of the pretrained GPT2 for content model
    # for name, param in collaborate_model.named_parameters():
    #     # we allow only user/item token embeddings to be trained
    #     if ('user_embeddings' not in name) and ('item_embeddings' not in name):
    #         param.requires_grad = False

    accelerator.print("-----Showing the Trainable Parameters-----")
    for name, param in collaborate_model.named_parameters():
        if param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))
    
    accelerator.print("-----Showing the Non-trainable Parameters-----")
    for name, param in collaborate_model.named_parameters():
        if not param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))

    # Set up the training details
    accelerator.print("-----Setting Up the Training Details-----")
    
    learning_rate = 1e-4
    batch_size_content_based = 4
    batch_size_collaborative_based_training = 32
    batch_size_collaborative_based_validation = 32
    num_fine_tuning_epochs = 200

    accelerator.print("learning_rate: ", learning_rate)
    accelerator.print("batch_size_content_based: ", batch_size_content_based)
    accelerator.print("batch_size_collaborative_based_training: ", batch_size_collaborative_based_training)
    accelerator.print("batch_size_collaborative_based_validation: ", batch_size_collaborative_based_validation)
    accelerator.print("num_fine_tuning_epochs: ", num_fine_tuning_epochs)

    # Create a data sampler for distributed training
    num_workers = 1 # We are using single GPU
    collaborative_based_train_data_loader = DataLoader(collaborative_train_data_gen, 
                                   batch_size=batch_size_collaborative_based_training,
                                   collate_fn=collaborative_train_data_gen.collate_fn,
                                   num_workers=num_workers)
    collaborative_based_val_data_loader = DataLoader(collaborative_val_data_gen, 
                                 batch_size=batch_size_collaborative_based_validation, 
                                 collate_fn=collaborative_val_data_gen.collate_fn,
                                 num_workers=num_workers)
    content_based_data_loader = DataLoader(content_data_gen, 
                                    batch_size=batch_size_content_based, 
                                    collate_fn=content_data_gen.collate_fn,
                                    num_workers=num_workers)
    accelerator.print("-----DataLoader Created-----")

    # Set the model to the training mode
    collaborate_model.train()
    collaborate_model.to(device)
    content_model.train()
    content_model.to(device)

    # Obtain the optimizer
    collaborative_based_optimizer = optim.Adam(collaborate_model.parameters(), lr=learning_rate)
    content_based_optimizer = optim.Adam(content_model.parameters(), lr=learning_rate)

    # Model, optimizer and data loader with accelerator
    collaborate_model, collaborative_based_optimizer, collaborative_based_train_data_loader = \
        accelerator.prepare(collaborate_model, collaborative_based_optimizer, collaborative_based_train_data_loader)
    content_model, content_based_optimizer, content_based_data_loader = \
        accelerator.prepare(content_model, content_based_optimizer, content_based_data_loader)

    # Initialize best_loss with infinity
    review_best_loss = float('inf')
    best_val_rec_loss = float('inf')
    best_recall_20 = float('inf') * -1
    best_recall_40 = float('inf') * -1
    best_NDCG_100 = float('inf') * -1
    best_sum = float('inf') * -1

    # The places to save the pre-training checkpoints
    accelerator.print(f"Content-based model: Weights will be saved to {content_based_model_save_path}")
    accelerator.print(f"Collaborative-based model: Weights will be saved to {collaborative_model_save_path}")

    # Finished doing the setup

    # ----------------------------------------------------------------
    # The fine tuning loop begins from here    
    # ----------------------------------------------------------------
    accelerator.print("-----Begin Rec GPT Fine Tuning-----")

    for epoch in range(num_fine_tuning_epochs):
        collaborate_model.train()

        train_rec_loss = 0
        regularize_total_loss = 0 
        
        accelerator.print(f'Epoch {epoch + 1}/{num_fine_tuning_epochs}')
        progress_bar = tqdm(collaborative_based_train_data_loader, desc=f"Epoch {epoch + 1}",
                            disable=not accelerator.is_local_main_process, ncols=80)
        
        for input_ids, target_mat, attention_mask, input_ids_main, graph_bc in progress_bar:
            collaborative_based_optimizer.zero_grad()

            input_ids = input_ids.to(device)
            target_mat = target_mat.to(device)
            attention_mask = attention_mask.to(device)
            input_ids_main = input_ids_main.to(device)
            graph_bc = graph_bc.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                content_embeds = torch.cat(
                    (accelerator.unwrap_model(content_model).base_model.embed(input_ids),
                    accelerator.unwrap_model(content_model).base_model.embed(input_ids_main)),
                    axis=1
                ).to(device)

            # Forward pass
            outputs = collaborate_model(input_ids, 
                                target_mat,
                                mapping_graph_bc=graph_bc,
                                attention_mask=attention_mask,
                                regularize=True,
                                lambda_V=lambda_V,
                                main_ids=input_ids_main,
                                content_embeds=content_embeds)
            rec_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            accelerator.backward(rec_loss)
            collaborative_based_optimizer.step()

            train_rec_loss += rec_loss.item()
            regularize_total_loss += regularize_loss.item()
            progress_bar.set_postfix({"Rec Loss": rec_loss.item()})

        # Gather the multinomial recommendation loss from different device
        thread_train_rec_loss = torch.tensor([train_rec_loss / len(collaborative_based_train_data_loader)]).to(device)
        gathered_train_rec_loss = accelerator.gather(thread_train_rec_loss)
        train_rec_loss = torch.mean(gathered_train_rec_loss)
        accelerator.print(f"Epoch {epoch + 1} - Rec Loss: {train_rec_loss:.4f}")

        # Gather the regularize loss from difference device
        thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(collaborative_based_train_data_loader)]).to(device)
        gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
        regularize_average_loss = torch.mean(gathered_regularize_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

        # Set the model to evaluation mode
        collaborate_model.eval()  
        val_rec_loss = 0
        cur_recall_20 = 0
        cur_recall_40 = 0
        cur_NDCG_100 = 0

        accelerator.wait_for_everyone()
        with torch.no_grad():
            for input_ids, train_mat, target_mat, attention_mask, graph_bc in collaborative_based_val_data_loader:
                # Move tensors to the correct device
                input_ids = input_ids.to(device)
                train_mat = train_mat.to(device)
                target_mat = target_mat.to(device)
                attention_mask = attention_mask.to(device)
                graph_bc = graph_bc.to(device)

                # Get item scores and rank them
                rec_loss, item_scores = collaborate_model(input_ids,
                                                target_mat,
                                                mapping_graph_bc=graph_bc,
                                                attention_mask=attention_mask)
                
                # Set score of interacted items to the lowest
                item_scores[train_mat > 0] = -float("inf")  

                # Calculate Recall@K and NDCG@K for each user
                target_mat = target_mat.cpu().numpy()
                item_scores = item_scores.cpu().numpy()
                val_rec_loss += rec_loss.item()
                cur_recall_20 += Recall_at_k(target_mat, item_scores, k=20, agg="sum")
                cur_recall_40 += Recall_at_k(target_mat, item_scores, k=40, agg="sum")
                cur_NDCG_100 += NDCG_at_k(target_mat, item_scores, k=100, agg="sum")

        # Calculate average Recall@K and NDCG@K for the validation set
        val_rec_loss /= len(collaborative_based_val_data_loader)
        cur_recall_20 /= len(collaborative_val_data_gen)
        cur_recall_40 /= len(collaborative_val_data_gen)
        cur_NDCG_100 /= len(collaborative_val_data_gen)
        cur_sum = cur_recall_20 + cur_recall_40 + cur_NDCG_100
    
        # Update the best metrics
        if val_rec_loss < best_val_rec_loss:
            best_val_rec_loss = val_rec_loss
        if cur_recall_20 > best_recall_20:
            best_recall_20 = cur_recall_20
        if cur_recall_40 > best_recall_40:
            best_recall_40 = cur_recall_40
        if cur_NDCG_100 > best_NDCG_100:
            best_NDCG_100 = cur_NDCG_100
        if cur_sum > best_sum:
            best_sum = cur_sum

            # Save user embeddings
            user_emb_path = os.path.join(collaborative_model_save_path, f"user_embeddings.pt")
            item_emb_path = os.path.join(collaborative_model_save_path, f"item_embeddings.pt")
            gpt_save_path = os.path.join(collaborative_model_save_path, f"collaborative_based_gpt2.bin")
            torch.save(accelerator.unwrap_model(collaborate_model).base_model.user_embeddings.state_dict(), user_emb_path)
            torch.save(accelerator.unwrap_model(collaborate_model).base_model.item_embeddings.state_dict(), item_emb_path)
            torch.save(accelerator.unwrap_model(collaborate_model).base_model.gpt2model.state_dict(), gpt_save_path)

        accelerator.print(f"Best model saved to {collaborative_model_save_path}")
        accelerator.print(f"Train Rec Loss: {train_rec_loss:.4f}")
        accelerator.print(f"Val Rec Loss: {val_rec_loss:.4f} / Best Val Rec Loss: {best_val_rec_loss:.4f}")
        accelerator.print(f"Cur Recall@20: {cur_recall_20:.4f} / Best Recall@20: {best_recall_20:.4f}")
        accelerator.print(f"Cur Recall@40: {cur_recall_40:.4f} / Best Recall@40: {best_recall_40:.4f}")
        accelerator.print(f"Cur NDCG@100: {cur_NDCG_100:.4f} / Best NDCG@100: {best_NDCG_100:.4f}")    

        if (epoch + 1) % 150 == 0:
            review_total_loss = 0
            regularize_total_loss = 0

            progress_bar = tqdm(content_based_data_loader, desc=f"Epoch {epoch + 1}",
                            disable=not accelerator.is_local_main_process, ncols=100)
            
            for input_ids_prompt, input_ids_main, attention_mask, graph_bc_prompt, graph_bc_combined in progress_bar:
                content_based_optimizer.zero_grad()

                input_ids_prompt = input_ids_prompt.to(device)
                input_ids_main = input_ids_main.to(device)
                attention_mask = attention_mask.to(device)
                graph_bc_prompt = graph_bc_prompt.to(device)
                graph_bc_combined = graph_bc_combined.to(device)

                accelerator.wait_for_everyone()
                with torch.no_grad():
                    rec_embeds = accelerator.unwrap_model(collaborate_model).\
                                base_model.embed(input_ids_prompt).to(device)

                # Forward pass of the content GPT
                outputs = content_model(input_ids_prompt,
                                        input_ids_main,
                                        mapping_graph_bc_prompt=graph_bc_prompt,
                                        mapping_graph_bc_combined=graph_bc_combined,
                                        labels_main=input_ids_main,
                                        attention_mask=attention_mask,
                                        regularize=True,
                                        lambda_V=lambda_V,
                                        collaborative_embeds=rec_embeds)
                review_loss = outputs[0]
                regularize_loss = outputs[1]

                # Backward pass and optimization
                accelerator.backward(review_loss)
                content_based_optimizer.step()

                review_total_loss += review_loss.item()
                regularize_total_loss += regularize_loss.item()
                progress_bar.set_postfix({"Review Loss": review_loss.item(),
                                        "Regularize Loss": regularize_loss.item()})

            # Gather the content LM loss from different device
            thread_review_average_loss = torch.tensor([review_total_loss / len(content_based_data_loader)]).to(device)
            gathered_review_average_loss = accelerator.gather(thread_review_average_loss)
            review_average_loss = torch.mean(gathered_review_average_loss)
            accelerator.print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")

            # Gather the regularize loss from different device
            thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(content_based_data_loader)]).to(device)
            gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
            regularize_average_loss = torch.mean(gathered_regularize_average_loss)
            accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

            # Check if the current loss is better than the best_loss
            if review_average_loss < review_best_loss:
                review_best_loss = review_average_loss

                # Save user embeddings in the main process
                user_emb_path = os.path.join(content_based_model_save_path, f"user_embeddings.pt")
                item_emb_path = os.path.join(content_based_model_save_path, f"item_embeddings.pt")
                gpt_save_path = os.path.join(content_based_model_save_path, f"content_based_gpt2.bin")
                torch.save(accelerator.unwrap_model(content_model).base_model.user_embeddings.state_dict(), user_emb_path)
                torch.save(accelerator.unwrap_model(content_model).base_model.item_embeddings.state_dict(), item_emb_path)
                torch.save(accelerator.unwrap_model(content_model).base_model.gpt2model.state_dict(), gpt_save_path)


if __name__ == "__main__":
    main()