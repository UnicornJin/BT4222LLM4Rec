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

from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Config

sys.path.append("libs")
from libs.modeling_gpt2 import GPT2ModelWithBC
from libs.tokenizer import TokenizerWithUserItemIDTokensBatch
from libs.data import RecommendationGPTTestGeneratorBatch
from libs.model import GPT4RecommendationBaseModel
from libs.model import CollaborativeGPTwithItemRecommendHead
from libs.util import Recall_at_k, NDCG_at_k

# +++++++++++++++++++++++++++++++++++++++
dataset = 'luxury'
lambda_V = 1.0

# Dataset Related file paths
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
test_matrix_path = os.path.join(data_root, "test_matrix.npz")


pre_train_checkpoint = os.path.join('./checkpoints', 'pretrain', dataset)
fine_tune_checkpoint = os.path.join('./checkpoints', 'finetune', dataset)

bt4222_gpt2_finetune_weights_path = os.path.join(fine_tune_checkpoint, "collaborative-based", "collaborative_based_gpt2.bin")
bt4222_gpt2_finetune_user_emb_path = os.path.join(fine_tune_checkpoint, "collaborative-based", "user_embeddings.pt")
bt4222_gpt2_finetune_item_emb_path = os.path.join(fine_tune_checkpoint, "collaborative-based", "item_embeddings.pt")

# Need to use the author's provided tokenizer, instead of the original one
provided_tokenizer_path = './provided_tokenizer'
provided_vocab_file = os.path.join(provided_tokenizer_path, "vocab_file.json")
provided_merges_file = os.path.join(provided_tokenizer_path, "merges.txt")

# The paths to save evaluation results, when you run by yourself
self_running_result_save_dir = './results'
results_save_path = os.path.join(self_running_result_save_dir, dataset)
if not os.path.exists(results_save_path):
    os.makedirs(results_save_path, exist_ok=True)
# +++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++
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

def main():
    print("-----Current Setting-----")
    print(f"dataset: {dataset}")
    print(f"lambda_V: {lambda_V}")

    device = "cuda"

    # Get the basic information of the dataset
    print("-----Dataset Info-----")
    with fsspec.open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
    num_users = meta_data["num_users"]
    num_items = meta_data["num_items"]
    print(f"Number of Users: {num_users}")
    print(f"Number of Items: {num_items}")


    # Load the tokenizer with user/item tokens
    print("-----Loading the Tokenizer-----")
    print(f"Loading pretrained tokenizer from {provided_tokenizer_path}")
    tokenizer = TokenizerWithUserItemIDTokensBatch(provided_vocab_file, provided_merges_file, num_users, num_items)

    # Load the testing data
    print("-----Loading the test data-----")
    mapping_graph_bc = np.ones((num_users + num_items, num_users + num_items))
    train_mat = load_npz(train_matrix_path)
    test_mat = load_npz(test_matrix_path)
    test_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, test_mat, mapping_graph_bc)
    
    # The config of the original GPT model needs to be extended, for fitting the tasks
    # The changing is simple, just add two attributes: num_users, num_items
    print("-----Begin Setting Up the Config-----")
    config = GPT2Config(**_config)
    config.num_users = num_users
    config.num_items = num_items

    '''
        Instantiate the pretrained GPT2 model
    '''
    print("-----Begin Instantiating the Pretrained GPT Model-----")
    
    gpt2model = GPT2ModelWithBC(config)    
    gpt2model.load_state_dict(torch.load(bt4222_gpt2_finetune_weights_path), strict=False)
    base_model = GPT4RecommendationBaseModel(config, gpt2model)
    base_model.user_embeddings.load_state_dict(torch.load(bt4222_gpt2_finetune_user_emb_path, map_location=device))
    base_model.item_embeddings.load_state_dict(torch.load(bt4222_gpt2_finetune_item_emb_path, map_location=device))
    rec_model = CollaborativeGPTwithItemRecommendHead(config, base_model)

    print("-----Create the DataLoader-----")
    batch_size = 256
    test_data_loader = DataLoader(test_data_gen,
                                  batch_size=batch_size,
                                  collate_fn=test_data_gen.collate_fn)

    rec_model.to(device)
    rec_model.eval()

    cur_recall_20 = 0
    cur_recall_40 = 0
    cur_NDCG_100 = 0

    print("-----Running the model-----")
    with torch.no_grad():
        for input_ids, train_mat, target_mat, attention_mask, graph_bc in test_data_loader:

            input_ids = input_ids.to(device)
            train_mat = train_mat.to(device)
            target_mat = target_mat.to(device)
            attention_mask = attention_mask.to(device)
            graph_bc = graph_bc.to(device)

            rec_loss, item_scores = rec_model(input_ids,
                                              target_mat,
                                              mapping_graph_bc=graph_bc,
                                              attention_mask=attention_mask)

            # Set score of interacted items to the lowest
            item_scores[train_mat > 0] = -float("inf")

            # Calculate Recall@K and NDCG@K for each user
            target_mat = target_mat.cpu().numpy()
            item_scores = item_scores.cpu().numpy()
            cur_recall_20 += Recall_at_k(target_mat, item_scores, k=20, agg="sum")
            cur_recall_40 += Recall_at_k(target_mat, item_scores, k=40, agg="sum")
            cur_NDCG_100 += NDCG_at_k(target_mat, item_scores, k=100, agg="sum")

    # Calculate average Recall@K and NDCG@K for the validation set
    cur_recall_20 /= len(test_data_gen)
    cur_recall_40 /= len(test_data_gen)
    cur_NDCG_100 /= len(test_data_gen)

    print(f"Final Testing Results:")
    print(f"Recall@20: {cur_recall_20:.4f}")
    print(f"Recall@40: {cur_recall_40:.4f}")
    print(f"NDCG@100: {cur_NDCG_100:.4f}")

    results_path = os.path.join(results_save_path, f"results.txt")
    with fsspec.open(results_path, "w") as f:
        f.write(f"Final Testing Results:\n")
        f.write(f"Recall@20: {cur_recall_20:.4f}\n")
        f.write(f"Recall@40: {cur_recall_40:.4f}\n")
        f.write(f"NDCG@100: {cur_NDCG_100:.4f}\n")

if __name__ == "__main__":
    main()
