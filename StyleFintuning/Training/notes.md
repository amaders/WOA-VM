Notes for Axolotl YAML SFT Config 


```
datasets:

  - path: /Users/amader/Desktop/Work/CuiLab/WOA_VM/StyleFintuning/Data/ax_data/train.jsonl #path to trian dataset
    type: #type is null for SFT

    chat_template: chatml #we are using ChatML instruction format for training

    message_property_mappings:  # our mapping is consistent with ChatML
      role: role
      content: content
    

    #define how to map role keys to roles (this is may be redundant, but whatever)
    roles:
      assistant:
        - assistant
      user:
        - user
```
