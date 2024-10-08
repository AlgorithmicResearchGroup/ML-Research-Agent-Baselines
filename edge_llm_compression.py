import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils import prune

OUTPUT_DIR = "edge_llm_compression_model"

def load_model():
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    return model, tokenizer

def compress_model(model):
    # 1. Pruning
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.3)
    
    # 2. Weight sharing
    def share_weights(layer1, layer2):
        layer2.weight = layer1.weight
    
    for i in range(1, len(model.transformer.h), 2):
        share_weights(model.transformer.h[i-1].attn.c_attn, model.transformer.h[i].attn.c_attn)
        share_weights(model.transformer.h[i-1].attn.c_proj, model.transformer.h[i].attn.c_proj)
    
    # 3. Reduce model width
    def reduce_width(module, reduction_factor=0.8):
        if isinstance(module, torch.nn.Linear):
            new_out_features = int(module.out_features * reduction_factor)
            new_weight = module.weight[:new_out_features, :]
            new_bias = module.bias[:new_out_features] if module.bias is not None else None
            
            new_linear = torch.nn.Linear(module.in_features, new_out_features, bias=module.bias is not None)
            new_linear.weight.data = new_weight.data
            if new_bias is not None:
                new_linear.bias.data = new_bias.data
            
            return new_linear
        return module

    for name, module in model.named_children():
        if isinstance(module, torch.nn.ModuleList):
            for i, layer in enumerate(module):
                module[i] = torch.nn.Sequential(*[reduce_width(m) for m in layer.children()])
    
    # 4. Remove some layers
    num_layers_to_remove = 2
    model.transformer.h = model.transformer.h[:-num_layers_to_remove]
    
    return model

def main():
    model, tokenizer = load_model()
    
    print("Original model size:", sum(p.numel() for p in model.parameters()) * 2 / 1024 / 1024, "MB")
    
    compressed_model = compress_model(model)
    
    print("Compressed model size:", sum(p.numel() for p in compressed_model.parameters()) * 2 / 1024 / 1024, "MB")
    
    # Save the compressed model
    compressed_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
if __name__ == "__main__":
    main()