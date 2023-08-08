from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch
import numpy

def save_model(path):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    # for name, params in model.named_parameters():
    #     print(f"name: {name}, params: {params.shape}")
    module_dict = {n: p.detach() for n, p in model.named_parameters()}
    modi_module_dict = {}

    for name, params in module_dict.items():
        if "wte" in name:
            name = name.replace("wte", "token_embed")
        if "wpe" in name:
            name = name.replace("wpe", "posi_embed")
        if "c_proj" in name:
            name = name.replace("c_proj", "project")
        if "c_fc" in name:
            name = name.replace("c_fc", "fc")
        
        if "c_attn" in name:
            names_ = name.split("c_attn")
            num_head = 12
            if "weight" in name:
                split_size = params.shape[1] // 3
                atten_size = split_size // num_head
                Q = params[:, :split_size]
                K = params[:, split_size:2*split_size]
                V = params[:, 2*split_size:]
                qs = Q.split(split_size=atten_size, dim=1)
                ks = K.split(split_size=atten_size, dim=1)
                vs = V.split(split_size=atten_size, dim=1)
                for i in range(num_head):
                    new_name = names_[0] + f"q_{i}" + names_[1]
                    modi_module_dict[new_name] = qs[i]
                for i in range(num_head):
                    new_name = names_[0] + f"k_{i}" + names_[1]
                    modi_module_dict[new_name] = ks[i]
                for i in range(num_head):
                    new_name = names_[0] + f"v_{i}" + names_[1]
                    modi_module_dict[new_name] = vs[i]
            if "bias" in name:
                split_size = params.shape[0] // 3
                atten_size = split_size // num_head
                Q = params[:split_size]
                K = params[split_size:2*split_size]
                V = params[2*split_size:]
                qs = Q.split(split_size=atten_size, dim=0)
                ks = K.split(split_size=atten_size, dim=0)
                vs = V.split(split_size=atten_size, dim=0)
                for i in range(num_head):
                    new_name = names_[0] + f"q_{i}" + names_[1]
                    modi_module_dict[new_name] = qs[i]
                for i in range(num_head):
                    new_name = names_[0] + f"k_{i}" + names_[1]
                    modi_module_dict[new_name] = ks[i]
                for i in range(num_head):
                    new_name = names_[0] + f"v_{i}" + names_[1]
                    modi_module_dict[new_name] = vs[i]
        else:
            modi_module_dict[name] = params

    for name, params in model.get_output_embeddings().named_parameters():
        modi_module_dict["lm_head"] = params.detach()
        # print(f'name: lm_head, params: {params.shape}')

    for name, params in modi_module_dict.items():
        npy_params = numpy.expand_dims(params.numpy(), axis=0)
        modi_module_dict[name] = npy_params
        print(f"name: {name}, params: {npy_params.shape}")

    numpy.savez_compressed(path, **modi_module_dict)
    # numpy.save(path, **modi_module_dict)

if __name__ == '__main__':
    save_model(path="../model/gpt2.npz")


    

    
