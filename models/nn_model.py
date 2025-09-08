import torch
from nnsight import NNsight, LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-14B"  # or the base model

llm = LanguageModel(model_id, device_map="auto")

print(llm)

with llm.trace("Love"):
    # user-defined code to access internal model components
    hs_all_love = llm.model.layers[1].input[0].save()
    output = llm.output.save()

print("Hidden State Logits: ",hs_all_love.shape)

with llm.trace("Hate"):
    # user-defined code to access internal model components
    hidden_states_love_ice_cream = llm.model.layers[1].input[0].save()
    output = llm.output.save()


print("Hidden State Logits 2: ",hidden_states_love_ice_cream.shape)


output_logits = output["logits"]
print("Model Output Logits: ",output_logits.shape)

# decode the final model output from output logits
max_probs, tokens = output_logits[0].max(dim=-1)
word = [llm.tokenizer.decode(tokens.cpu()[-1])]
print("Model Output: ", word[0])



