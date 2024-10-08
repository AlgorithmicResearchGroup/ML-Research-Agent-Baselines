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
    
    # 2. Weight Sharing
    def share_weights(layer1, layer2):
        layer2.weight = layer1.weight
    
    layers = model.gpt_neox.layers
    for i in range(1, len(layers), 2):
        # Access the attention modules in GPT-NeoX layers
        share_weights(layers[i-1].attention.query_key_value, layers[i].attention.query_key_value)
        share_weights(layers[i-1].attention.dense, layers[i].attention.dense)
    
    # 3. Reduce Model Width
    def reduce_width(module, reduction_factor=0.8):
        if isinstance(module, torch.nn.Linear):
            new_out_features = int(module.out_features * reduction_factor)
            new_weight = module.weight[:new_out_features, :]
            new_bias = module.bias[:new_out_features] if module.bias is not None else None
            
            new_linear = torch.nn.Linear(module.in_features, new_out_features, bias=module.bias is not None)
            new_linear.weight.data = new_weight.data.clone()
            if new_bias is not None:
                new_linear.bias.data = new_bias.data.clone()
            
            return new_linear
        return module

    # Apply reduce_width to the linear layers in each GPT-NeoX block
    for layer in layers:
        # Reduce the width of attention query_key_value and dense layers
        layer.attention.query_key_value = reduce_width(layer.attention.query_key_value)
        layer.attention.dense = reduce_width(layer.attention.dense)
        # Reduce the width of feed-forward layers
        layer.mlp.dense_h_to_4h = reduce_width(layer.mlp.dense_h_to_4h)
        layer.mlp.dense_4h_to_h = reduce_width(layer.mlp.dense_4h_to_h)
    
    # 4. Remove Some Layers
    num_layers_to_remove = 2
    model.gpt_neox.layers = torch.nn.ModuleList(layers[:-num_layers_to_remove])
    
    return model

def main():
    model, tokenizer = load_model()
    print(f"Model type: {type(model)}")
    
    print("Original model size:", sum(p.numel() for p in model.parameters()) * 2 / 1024 / 1024, "MB")
    
    compressed_model = compress_model(model)
    
    print("Compressed model size:", sum(p.numel() for p in compressed_model.parameters()) * 2 / 1024 / 1024, "MB")
    
    # Save the compressed model
    compressed_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
if __name__ == "__main__":
    main()
