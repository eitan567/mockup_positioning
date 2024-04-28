import torch
# from model import Generator  # Import the Generator from model.py
# from PIL import Image
# import torchvision.transforms as transforms

# # Load the pre-trained model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Generator().to(device)
# model.load_state_dict(torch.load('ai_models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth', map_location=device))
# model.eval()

# # Function to upscale an image
# def upscale_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((256, 256)),  # Resize to the input size expected by the model
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     input_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output_tensor = model(input_tensor)
    
#     output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
#     output_image.save('upscaled_image.png')

# # Upscale an example image
# upscale_image(r'pictures\1.png')


import torch
from model import RRDBNet  # This architecture is used in some ESRGAN implementations

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'ai_models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'  # Update this path to where you've saved the model
# model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(device)
# Create an instance of RRDBNet without the 'scale' parameter
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Example function to upscale an image
def upscale_image(image_path):
    from PIL import Image
    import torchvision.transforms as transforms
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # Adjust this depending on your needs
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    output_image.save('upscaled_image.png')

# Use the function with your image
upscale_image(r'pictures\1.png')
