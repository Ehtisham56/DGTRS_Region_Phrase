import torch
from PIL import Image
from model.dgtrs_longclip import DGTRSLongCLIP
from model import longclip

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing model architecture...")
    model = DGTRSLongCLIP(longclip_base_model="ViT-B/16").to(device)

    ckpt_path = "checkpoints/best_model.pt"
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Processing image...")
    # Get standard CLIP preprocess from longclip
    _, preprocess = longclip.load_from_clip("ViT-B/16", device="cpu", jit=False)
    
    image_path = "./img/demo.jpg"
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    print("Processing text...")
    texts = [
        "Numerous students are walking in the green pass in this campus.", 
        "These buildings belong to the school buildings.", 
        "There are many residential areas near the school.",
        "A photo of a dog."
    ]
    text_tokens = longclip.tokenize(texts).to(device)

    print("Running inference...")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        logit_scale = model.backbone.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * image_features @ text_features.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(f"\nImage: {image_path}")
    print("-" * 50)
    for i, text in enumerate(texts):
        print(f"Text '{text}' -> Probability: {probs[0][i]:.4f}")

if __name__ == "__main__":
    main()
