### How to use the data processing scripts

This folder contains all the scripts you need to convert the Amazon Review Dataset to the format our experiment needs.

The dataset can be found from: https://nijianmo.github.io/amazon/index.html 

You need to download a category of data. For our experiment, we are using "luxury". 

Please not that we are just using the "smaller" K-core subset dataset, for the tutorial. 

Once you choose a dataset, download: 
- The corresponding "metadata" (from the original dataset)
- The "5-core" reviews (from the smaller subset)

You will get something like: 
- `Luxury_Beauty_5.json.gz`
- `meta_Luxury_Beauty.json.gz`

Put these two files under `raw_data` folder. And run the scripts one by one:
- `python data_preprocess_amazon.py`
- `python data_preprocessing.py`
- `python data_pkl.py`

Then you will get the dataset required for the pre-training, fine-tuning, and evaluation.

If you want to use other categories of dataset, when you run the scripts, remember to change the corresponding variables. 

(If the dataset is too small, it may have some problems. e.g. the appliances dataset only has 3 items after our pre-processing. You better choose some dataset which contains around 10,000 to 40,000 reviews.)

For the details of what each script is doing, please refer to the scripts, they contains comments that explaining the steps. 