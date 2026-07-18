from huggingface_hub import snapshot_download
import os

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

print("Downloading DINOv3 ViT-L (~1.2 GB)...")
p1 = snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m')
print(f'DINOv3 at: {p1}')

print("\nDownloading OlmoEarth v1-Base (~830 MB)...")
p2 = snapshot_download('allenai/OlmoEarth-v1-Base')
print(f'OlmoEarth at: {p2}')

# Print the top-level dirs you need to upload to Drive
print(f'\n=== UPLOAD THESE TO GOOGLE DRIVE ===')
print(f'  {os.path.dirname(os.path.dirname(p1))}')
print(f'  {os.path.dirname(os.path.dirname(p2))}')