from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
# import torch.nn as nn
# import torch
import time
# print(torch.cuda.is_available())

gpt2_config = GPT2Config.from_pretrained('gpt2')
# print(gpt2_config)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')


# for name, params in gpt2_model.named_parameters():
#     print(f'name: {name}, params: {params.shape}')

# for name, params in gpt2_model.get_output_embeddings().named_parameters():
#     print(f'name: lm_head, params: {params.shape}')
# print(gpt2_config)

text = "Who is the Trump?"
start = time.time()
ids_input = tokenizer.encode(text, return_tensors='pt').to('cuda')
print(ids_input)
ids_output = gpt2_model.generate(ids_input, do_sample=True, top_k=50, top_p=0.95, max_length=100)
print(tokenizer.decode(ids_output[0]))
end = time.time()
print(f"inference time: {end - start:.4f}s")



# # ----------------- inference ----------------------
# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='gpt2')
# set_seed(42)
# outputs = generator("The white man worked as a", max_length=100, num_return_sequences=5)
# for output in outputs:
#     print(output['generated_text'])