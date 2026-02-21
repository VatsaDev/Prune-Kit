# manually select any layers you want to prune by index
# should work for any model of the HF format
# ex model has A ayers, prune range [0-(A-1)]* for all layers

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

# CONFIG

model_to_prune = "checkpoints/your_model/"
out_path = "checkpoints/out", # anything works

text_chop=[4, 10]
vis_chop=[9, 10]
deepstack=[1, 5, 8] # make sure to reset these anytime you do a vision prune, otherwise you may end up with a layer of pure noise/NaN, silently corrupts training, I always used [2nd layer, mid layer, 2nd last layer]

def get_param_count(model):
    return sum(p.numel() for p in model.parameters()) / 1e9

def prune_mod(model_id, output_path, text_remove_idx, vis_remove_idx, new_deepstack_idx):
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # dont think this should cause any issues, NOTE maybe abalate any fp32 master weight stuff for any gains?
        device_map="cpu",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Initial Size: {get_param_count(model):.3f}B")

    lang_layers = model.model.language_model.layers
    model.model.language_model.layers = torch.nn.ModuleList([
        l for i, l in enumerate(lang_layers) if i not in text_remove_idx
    ])

    vis_blocks = model.model.visual.blocks
    model.model.visual.blocks = torch.nn.ModuleList([
        b for i, b in enumerate(vis_blocks) if i not in vis_remove_idx
    ])

    model.config.text_config.num_hidden_layers = len(model.model.language_model.layers)
    model.config.vision_config.depth = len(model.model.visual.blocks)
    model.config.vision_config.deepstack_visual_indexes = new_deepstack_idx

    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(model.model.language_model.layers)

    print(f"Final Size: {get_param_count(model):.3f}B")
    print(f"Layers: {len(model.model.language_model.layers)} Text / {len(model.model.visual.blocks)} Vis")

    model.save_pretrained(output_path, safe_serialization=False)
    processor.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    if hasattr(model, 'generation_config'):
        model.generation_config.save_pretrained(output_path)

if __name__ == "__main__":
    prune_qwen_vl(
        model_id=model_to_prune,
        output_path=out_path,
        text_remove_idx=text_chop,
        vis_remove_idx=vis_chop,
        new_deepstack_idx=deepstack
    )
