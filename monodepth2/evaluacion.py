import os
import glob
import argparse
from PIL import Image
from torchvision import transforms
from networks import ResnetEncoder, DepthDecoder
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


def load_model(model_path, device):
    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc)

    encoder_path = os.path.join(model_path, "encoder.pth")
    decoder_path = os.path.join(model_path, "depth.pth")

    encoder.load_state_dict(torch.load(encoder_path, map_location=device), strict=False)
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device), strict=False)

    encoder.to(device)
    depth_decoder.to(device)

    encoder.eval()
    depth_decoder.eval()

    return encoder, depth_decoder

def predict_depth(encoder, depth_decoder, input_image, device):
    with torch.no_grad():
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
    return disp_resized

def evaluate_images(image_folder, model_path, output_folder, device):
    encoder, depth_decoder = load_model(model_path, device)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = glob.glob(os.path.join(image_folder, '*'))
    for idx, img_path in enumerate(image_paths):
        input_image = Image.open(img_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        disp_resized = predict_depth(encoder, depth_decoder, input_image, device)
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        
        output_image_path = os.path.join(output_folder, f"depth_{idx}.png")
        Image.fromarray((disp_resized_np * 255).astype('uint8')).save(output_image_path)
        print(f"Processed {img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MonoDepth2 evaluation script')
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--model_path', type=str, help='Path to the folder containing model weights')
    parser.add_argument('--output_folder', type=str, help='Path to the folder to save output depth maps')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_images(args.image_folder, args.model_path, args.output_folder, device)
