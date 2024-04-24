import streamlit as st
import torch
import torchvision
from PIL import Image
from utils import load_model_from_checkpoint
from utils import get_transforms
import streamlit as st
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Streamlit arguments")
    parser.add_argument("--model_path", type=str, help="Model path")
    return parser.parse_args()


# Parse command-line arguments
args = parse_args()
model_path = args.model_path


device = torch.device("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_built() else "cpu")

model = load_model_from_checkpoint(model_path)
model = model.to(device)
COLORSPACE = 'RGB'
RESOLUTION = ((128,128))
transform_style, transform_source = get_transforms(COLORSPACE, RESOLUTION)



# Convert the tensor values to the range [0, 255] and type `uint8` (required for image saving)
transform_to_image = transforms.Compose([
    transforms.Lambda(lambda x: x * 255),  # Scale from [0, 1] to [0, 255]
    transforms.Lambda(lambda x: x.byte())  # Convert to uint8
])


def main():



    st.title("UNetAdaiN demo")

   
  
    # Upload images

    col1, col2 = st.columns(2)
    col1.subheader('Upload original image')
    image1 = col1.file_uploader("", type=["jpg", "png", "jpeg"], key='source')
    if image1 is not None:
            source_image = Image.open(image1)
            if source_image.mode!='RGB':
                  source_image = source_image.convert('RGB')
            source_pt = transform_source(source_image)
            source_transformed = torchvision.transforms.functional.to_pil_image(source_pt)
            col1.image(source_transformed, caption="", use_column_width=True)

    col2.subheader('Upload style reference')
    image2 = col2.file_uploader("", type=["jpg", "png", "jpeg"], key='style')

    if image2 is not None:
            style_image = Image.open(image2)
            if style_image.mode!='RGB':
                  style_image = style_image.convert('RGB')
            style_pt = transform_style(style_image)
            style_transformed = torchvision.transforms.functional.to_pil_image(style_pt)
            col2.image(style_transformed, caption="", use_column_width=True)


    st.header('Recreated Image')
    if image1 and image2:
        _,recreated_image = model(source_pt.unsqueeze(0).to(device),
                        style_pt.unsqueeze(0).to(device))
        ## save recreated image to file
        print(recreated_image.shape, type(recreated_image))

        # Apply the transformations
        tensor = transform_to_image(recreated_image.squeeze(0))

        # Convert tensor to PIL image
        image = transforms.ToPILImage()(tensor)

        # Save the image to disk
        image.save("reconstruction.png")


        recreated_image= torchvision.transforms.functional.to_pil_image(recreated_image[0]) ## maybe problem here
        st.image(recreated_image, caption="", use_column_width=True)


    # Display some other image on the other half of the page
    #display_other_image(output_image)



if __name__ == "__main__":
    main()

