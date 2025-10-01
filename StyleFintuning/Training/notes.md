Notes for Axolotl YAML SFT Config 

Steps to train:
(all using .venv)
1. run either StyleFintuning/Data/dataset_prep_no_reasoning.ipynb or StyleFintuning/Data/dataset_prep_reasoning.ipynb to create the datasets
2. point config.yaml to those datasets
3. axolotl train StyleFintuning/Training/config.yaml