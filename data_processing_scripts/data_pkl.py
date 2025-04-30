import os
import pickle
import re

# ------------------------------------------------
# This script is to show the data processing of 
# BT4222 LLM for Recommendation Example Code
#
# This file contains the processing for item and categories information
# 
# The script is based on project 
# 'LLM4REC' https://github.com/anord-wang/LLM4REC
#
# Edition: 2025.05.01 by Jin Yuze
# ------------------------------------------------

dataset = 'luxury'

data_root = os.path.join("dataset", dataset)
item_text_root = os.path.join(data_root, "item_texts")
brand_pkl_path = os.path.join(item_text_root, "brand.pkl")
categories_pkl_path = os.path.join(item_text_root, "categories.pkl")

# Load data from 'brand.pkl'
with open(brand_pkl_path, 'rb') as file:
    brand_data = pickle.load(file)

# Load data from 'categories.pkl'
with open(categories_pkl_path, 'rb') as file:
    categories_data = pickle.load(file)

# ------------------------------------------------
# Process Loaded Data
brand_based_texts = {}
for text in brand_data:
    brand = text[1].strip()
    user = text[0]
    match = re.search(r'item_\d+', text[0])
    if match:
        extracted_item_id = match.group()
        if brand not in brand_based_texts:
            brand_based_texts[brand] = []
        brand_based_texts[brand].append(extracted_item_id)

category_based_texts = {}

for text in categories_data:
    categories = text[1].split(", ")
    # print('categories', categories)
    user = text[0]
    match = re.search(r'item_\d+', text[0])
    if match:
        extracted_item_id = match.group()
        for category in categories:
            if category.lower() != 'beauty':
                if category not in category_based_texts:
                    category_based_texts[category] = []
                category_based_texts[category].append(extracted_item_id)

# Save processed data to new files
new_item_text_root = os.path.join(data_root, "new_item_texts")
if not os.path.exists(new_item_text_root):
    os.makedirs(new_item_text_root)

item_texts = ["brand_extension", "categories_extension"]
item_texts = {item_text: [] for item_text in item_texts}

# Save 'brand_based_texts' and 'category_based_texts'
for category, texts in [("brand", brand_based_texts), ("categories", category_based_texts)]:
    if category == "brand":
        for brand, item_ids in texts.items():
            combined_text = ", ".join(item_ids)
            item_texts["brand_extension"].append([f"These item {combined_text} has the same brand:", f" {brand}"])
    if category == "categories":
        for categories, item_ids in texts.items():
            combined_text = ", ".join(item_ids)
            item_texts["categories_extension"].append([f"These items {combined_text} are all in the category:", f" {categories}"])

for name, item_text in item_texts.items():
    text_filepath = os.path.join(item_text_root, f"{name}.txt")
    pkl_filepath = os.path.join(item_text_root, f"{name}.pkl")

    with open(text_filepath, "w") as file:
        file.write("\n".join(
            ["".join([prompt, main]) for prompt, main in item_text]
        ))

    with open(pkl_filepath, "wb") as file:
        pickle.dump(item_text, file)
