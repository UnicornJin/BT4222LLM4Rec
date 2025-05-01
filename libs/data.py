import random
import fsspec
import pickle

import torch
from torch.utils.data import Dataset
import scipy.io


# ------------------------------------------------
# This script is to show the data generator of 
# BT4222 LLM for Recommendation Example Code
#
# This file contains the steps to convert data 
# into GPT training format prompts 
# 
# The script is based on project 
# 'LLM4REC' https://github.com/anord-wang/LLM4REC
#
# Edition: 2025.05.01 by Jin Yuze
# ------------------------------------------------

# Used in: Training
# Target: Generate samples in the format: "<user_i> has interacted with <item_a> <item_b> ..."
# Then model can first learn the user/item collaborative embedding, understanding the scenario.
class CollaborativeGPTGeneratorBatch(Dataset):
    """
    Dataset class for generating 
    collaborative-based GPT model input batches.

    Args:
        tokenizer (TokenizerWithUserItemIDTokensBatch):
            Custom tokenizer instance.
        train_mat (np.ndarray): 
            Matrix of user-item interactions.
        max_length (int, optional): 
            Maximum length of the encoded sequences. 
            Defaults to 1024.
    """

    def __init__(self, tokenizer, train_mat, mapping_graph_bc, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_mat = train_mat
        self.mapping_graph_bc = mapping_graph_bc
        self.max_length = max_length
        self.num_users, self.num_items = train_mat.shape
        self.vocab_size = 50257

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        
        # Tokenize the prompt
        prompt = f"user_{idx} has interacted with"

        return prompt, self.train_mat.getrow(idx).nonzero()[1]

    def get_bc_from_mapping(self, input_ids_1, input_ids_2, mapping_graph_bc):
        batch_size, seq_length_1 = input_ids_1.size()
        seq_length_2 = input_ids_2.size(1)

        inputs_graph_bc = torch.zeros((batch_size, seq_length_1, seq_length_2))

        for batch_idx in range(batch_size):
            for i in range(seq_length_1):
                for j in range(seq_length_2):
                    is_i_user_or_item = (input_ids_1[batch_idx, i] >= self.vocab_size)
                    is_j_user_or_item = (input_ids_2[batch_idx, j] >= self.vocab_size)

                    if is_i_user_or_item and is_j_user_or_item:
                        adjusted_i = input_ids_1[batch_idx, i] - self.vocab_size
                        adjusted_j = input_ids_2[batch_idx, j] - self.vocab_size
                        inputs_graph_bc[batch_idx, i, j] = mapping_graph_bc[adjusted_i, adjusted_j]

        return inputs_graph_bc

    # This is called by Dataloader, for the encoded and padded prompt IDs, main IDs, and attention masks
    def collate_fn(self, batch):
        """
        Custom collate function to encode and pad the batch of texts.

        Args:
            batch (List[Tuple[str, np.ndarray]]): 
                List of tuples containing the prompt and item IDs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Tuple containing the encoded and padded prompt IDs,
                main IDs, and attention masks.
        """
        prompt_texts, item_ids = zip(*batch)

        # Encode and pad the prompt and main texts
        encoded_prompt = self.tokenizer.encode_batch(prompt_texts)
        item_tokens = [" ".join(["item_" + str(item_id) for item_id in ids]) for ids in item_ids]
        encoded_main = self.tokenizer.encode_batch(item_tokens)

        # Get the prompt IDs, main IDs, and attention masks
        prompt_ids = torch.tensor(encoded_prompt[0])
        main_ids = torch.tensor(encoded_main[0])
        attention_mask = torch.cat((torch.tensor(encoded_prompt[1]), torch.tensor(encoded_main[1])), dim=1)

        # Truncate main IDs and attention mask if total length exceeds the maximum length
        total_length = prompt_ids.size(1) + main_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            main_ids = main_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        graph_bc_prompt = self.get_bc_from_mapping(prompt_ids, prompt_ids, self.mapping_graph_bc)
        combined_ids = torch.cat((prompt_ids, main_ids), dim=1)
        graph_bc_combined = self.get_bc_from_mapping(main_ids, combined_ids, self.mapping_graph_bc)

        return prompt_ids, main_ids, attention_mask, graph_bc_prompt, graph_bc_combined

# Used in: Training, Finetuning
# Contains: 
# 1. User's review to items, you can find the processed reviews under dataset/luxury/user_item_texts
# 2. Each item's description, you can find them under dataset/luxury/user_item_texts
# These are natural language based information, 
# thus the LLM can understand the high level meaning of items & user interactions,
# to assist better for the recommendation.
class UserItemContentGPTDatasetBatch(Dataset):
    """
    Dataset class for generating user-item 
    content-based GPT model input batches.

    Args:
        tokenizer (TokenizerWithUserItemIDTokensBatch): 
            Custom tokenizer instance.
        filepath (str): 
            Path to the pickle file containing the descriptions.
        max_length (int, optional): 
            Maximum length of the encoded sequences. Defaults to 1024.
    """

    def __init__(self, tokenizer, filepath_list, mapping_graph_bc, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mapping_graph_bc = mapping_graph_bc
        self.vocab_size = 50257
        self.data = []

        # Load descriptions from text file
        for filepath in filepath_list:
            assert filepath.endswith(".pkl"), "we need to load from a pickle file"
            with fsspec.open(filepath, 'rb') as file:
                self.data.extend(pickle.load(file))
            print(len(self.data))

    def __len__(self):
        return len(self.data)

    def get_bc_from_mapping(self, input_ids_1, input_ids_2, mapping_graph_bc):
        batch_size, seq_length_1 = input_ids_1.size()
        seq_length_2 = input_ids_2.size(1)

        inputs_graph_bc = torch.zeros((batch_size, seq_length_1, seq_length_2))

        for batch_idx in range(batch_size):
            for i in range(seq_length_1):
                for j in range(seq_length_2):
                    is_i_user_or_item = (input_ids_1[batch_idx, i] >= self.vocab_size)
                    is_j_user_or_item = (input_ids_2[batch_idx, j] >= self.vocab_size)

                    if is_i_user_or_item and is_j_user_or_item:
                        adjusted_i = input_ids_1[batch_idx, i] - self.vocab_size
                        adjusted_j = input_ids_2[batch_idx, j] - self.vocab_size
                        inputs_graph_bc[batch_idx, i, j] = mapping_graph_bc[adjusted_i, adjusted_j]

        return inputs_graph_bc

    def __getitem__(self, idx):

        # Get the prompt and main texts
        prompt_text, main_text = self.data[idx][0], self.data[idx][1]

        return prompt_text, main_text

    def collate_fn(self, batch):
        """
        Custom collate function to encode and pad the batch of texts.

        Args:
            batch (List[Tuple[str, str]]): 
                List of tuples containing the prompt and main texts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Tuple containing the encoded and padded prompt IDs,
                main IDs, and attention masks.
        """
        prompt_texts, main_texts = zip(*batch)

        # Encode and pad the prompt and main texts
        encoded_prompt = self.tokenizer.encode_batch(prompt_texts)
        encoded_main = self.tokenizer.encode_batch(main_texts)

        # Get the prompt IDs, main IDs, and attention masks
        prompt_ids = torch.tensor(encoded_prompt[0])
        main_ids = torch.tensor(encoded_main[0])
        
        # check the length
        prompt_attention_mask = torch.tensor(encoded_prompt[1])
        if prompt_ids.size(1) > self.max_length:
            prompt_ids = prompt_ids[:, :(self.max_length - 8)]
            prompt_attention_mask = prompt_attention_mask[:, :(self.max_length - 8)]
        
        attention_mask = torch.cat((prompt_attention_mask, torch.tensor(encoded_main[1])), dim=1)

        # Truncate main IDs and attention mask if total length exceeds the maximum length
        total_length = prompt_ids.size(1) + main_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            main_ids = main_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        graph_bc_prompt = self.get_bc_from_mapping(prompt_ids, prompt_ids, self.mapping_graph_bc)
        combined_ids = torch.cat((prompt_ids, main_ids), dim=1)
        graph_bc_combined = self.get_bc_from_mapping(main_ids, combined_ids, self.mapping_graph_bc)

        return prompt_ids, main_ids, attention_mask, graph_bc_prompt, graph_bc_combined

# Used in: Finetuning
# The prompt is changed to "<user_i> has interacted with <item_a> <item_b> ..., user_{idx} will interact with "
# from here, we start to require the model to do prediction, and judging the correctness of the prediction.
class RecommendationGPTTrainGeneratorBatch(Dataset):
    """
    Dataset class for generating recommendation GPT input batches.

    Args:
        tokenizer (TokenizerWithUserItemIDTokensBatch):
            Custom tokenizer instance.
        train_mat (np.ndarray): 
            Matrix of user-item interactions.
        max_length (int, optional): 
            Maximum length of the encoded sequences. 
            Defaults to 1024.
        predict_ratio (float, optional):
            The percentage of items to predict for each user (default: 0.2).
    """

    def __init__(self,
                 tokenizer,
                 train_mat,
                 mapping_graph_bc,
                 max_length=1024,
                 predict_ratio=0.2,
                 shuffle=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_mat = train_mat
        self.max_length = max_length
        self.num_users, self.num_items = train_mat.shape
        self.predict_ratio = predict_ratio
        self.shuffle = shuffle
        self.mapping_graph_bc = mapping_graph_bc
        self.vocab_size = 50257

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        # Get past item interactions for the user
        past_interactions = self.train_mat.getrow(idx).nonzero()[1]

        # Randomly mask 20% of the items as input
        num_items_to_mask = max(1, int(len(past_interactions) * 0.2))
        masked_items = random.sample(past_interactions.tolist(), num_items_to_mask)

        # Generate the input and target interactions
        input_interactions = [item if item not in masked_items else None for item in past_interactions]
        if self.shuffle:
            random.shuffle(input_interactions)
        target_interactions = past_interactions

        # Tokenize the input and create the target matrix
        input_prompt = f"user_{idx} has interacted with {' '.join(['item_' + str(item_id) for item_id in input_interactions if item_id is not None])}"
        input_prompt += f", user_{idx} will interact with"

        target_matrix = torch.zeros(self.num_items, dtype=torch.float32)
        target_matrix[target_interactions] = 1.0
        item_ids = target_matrix.nonzero()[0]

        return input_prompt, target_matrix, item_ids

    def get_bc_from_mapping(self, input_ids_1, input_ids_2, mapping_graph_bc):
        batch_size, seq_length_1 = input_ids_1.size()
        seq_length_2 = input_ids_2.size(1)

        inputs_graph_bc = torch.zeros((batch_size, seq_length_1, seq_length_2))

        for batch_idx in range(batch_size):
            for i in range(seq_length_1):
                for j in range(seq_length_2):
                    is_i_user_or_item = (input_ids_1[batch_idx, i] >= self.vocab_size)
                    is_j_user_or_item = (input_ids_2[batch_idx, j] >= self.vocab_size)

                    if is_i_user_or_item and is_j_user_or_item:
                        adjusted_i = input_ids_1[batch_idx, i] - self.vocab_size
                        adjusted_j = input_ids_2[batch_idx, j] - self.vocab_size
                        inputs_graph_bc[batch_idx, i, j] = mapping_graph_bc[adjusted_i, adjusted_j]

        return inputs_graph_bc

    def collate_fn(self, batch):
        """
        Custom collate function to encode and pad the batch of texts.

        Args:
            batch (List[Tuple[str, torch.Tensor]]):
                List of tuples containing the prompt and target matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Tuple containing the encoded and padded prompt IDs,
                target matrix, and attention mask.
        """
        prompt_texts, target_matrices, item_ids = zip(*batch)

        # Encode and pad the prompt and main texts
        encoded_prompt = self.tokenizer.encode_batch(prompt_texts)
        target_matrices = torch.cat([matrix.unsqueeze(0) for matrix in target_matrices])
        item_tokens = [" ".join(["item_" + str(item_id) for item_id in ids]) for ids in item_ids]
        encoded_main = self.tokenizer.encode_batch(item_tokens)

        # Get the prompt IDs, target matrices, and attention masks
        prompt_ids = torch.tensor(encoded_prompt[0])
        main_ids = torch.tensor(encoded_main[0])
        attention_mask = torch.tensor(encoded_prompt[1])

        # Truncate prompt IDs and attention mask if total length exceeds the maximum length
        total_length = prompt_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            prompt_ids = prompt_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        graph_bc = self.get_bc_from_mapping(prompt_ids, prompt_ids, self.mapping_graph_bc)

        return prompt_ids, target_matrices, attention_mask, main_ids, graph_bc

# Used in: Finetuning, Evaluation
# This is for Test dataset, to judge the performance of model.
class RecommendationGPTTestGeneratorBatch(Dataset):
    """
    Dataset class for generating recommendation GPT input batches.

    Args:
        tokenizer (TokenizerWithUserItemIDTokensBatch):
            Custom tokenizer instance.
        train_mat (np.ndarray): 
            Matrix of user-item interactions.
        max_length (int, optional): 
            Maximum length of the encoded sequences. 
            Defaults to 1024.
        predict_ratio (float, optional):
            The percentage of items to predict for each user (default: 0.2).
    """

    def __init__(self,
                 tokenizer,
                 train_mat,
                 test_mat,
                 mapping_graph_bc,
                 max_length=1024,
                 predict_ratio=0.2,
                 shuffle=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_mat = train_mat
        self.test_mat = test_mat
        self.max_length = max_length
        self.num_users, self.num_items = train_mat.shape
        self.predict_ratio = predict_ratio
        self.shuffle = shuffle
        self.mapping_graph_bc = mapping_graph_bc
        self.vocab_size = 50257

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        # Get past item interactions for the user
        input_interactions = self.train_mat.getrow(idx).nonzero()[1]
        if self.shuffle:
            random.shuffle(input_interactions)

        # Tokenize the input and create the target matrix
        input_prompt = f"user_{idx} has interacted with {' '.join(['item_' + str(item_id) for item_id in input_interactions])}"
        input_prompt += f", user_{idx} will interact with"

        # Obtain the training items
        train_interactions = self.train_mat.getrow(idx).nonzero()[1]
        train_matrix = torch.zeros(self.num_items, dtype=torch.float32)
        train_matrix[train_interactions] = 1.0

        # Obtain the val/test items
        target_interactions = self.test_mat.getrow(idx).nonzero()[1]
        target_matrix = torch.zeros(self.num_items, dtype=torch.float32)
        target_matrix[target_interactions] = 1.0

        return input_prompt, train_matrix, target_matrix

    def get_bc_from_mapping(self, input_ids_1, input_ids_2, mapping_graph_bc):
        batch_size, seq_length_1 = input_ids_1.size()
        seq_length_2 = input_ids_2.size(1)

        inputs_graph_bc = torch.zeros((batch_size, seq_length_1, seq_length_2))

        for batch_idx in range(batch_size):
            for i in range(seq_length_1):
                for j in range(seq_length_2):
                    is_i_user_or_item = (input_ids_1[batch_idx, i] >= self.vocab_size)
                    is_j_user_or_item = (input_ids_2[batch_idx, j] >= self.vocab_size)

                    if is_i_user_or_item and is_j_user_or_item:
                        adjusted_i = input_ids_1[batch_idx, i] - self.vocab_size
                        adjusted_j = input_ids_2[batch_idx, j] - self.vocab_size
                        inputs_graph_bc[batch_idx, i, j] = mapping_graph_bc[adjusted_i, adjusted_j]

        return inputs_graph_bc

    def collate_fn(self, batch):
        """
        Custom collate function to encode and pad the batch of texts.

        Args:
            batch (List[Tuple[str, torch.Tensor]]): 
                List of tuples containing the prompt and target matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Tuple containing the encoded and padded prompt IDs,
                target matrix, and attention mask.
        """
        prompt_texts, train_matrices, target_matrices = zip(*batch)

        # Encode and pad the prompt and main texts
        encoded_prompt = self.tokenizer.encode_batch(prompt_texts)
        train_matrices = torch.cat([matrix.unsqueeze(0) for matrix in train_matrices])
        target_matrices = torch.cat([matrix.unsqueeze(0) for matrix in target_matrices])

        # Get the prompt IDs, target matrices, and attention masks
        prompt_ids = torch.tensor(encoded_prompt[0])
        attention_mask = torch.tensor(encoded_prompt[1])

        # Truncate prompt IDs and attention mask if total length exceeds the maximum length
        total_length = prompt_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            prompt_ids = prompt_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        graph_bc = self.get_bc_from_mapping(prompt_ids, prompt_ids, self.mapping_graph_bc)

        return prompt_ids, train_matrices, target_matrices, attention_mask, graph_bc
