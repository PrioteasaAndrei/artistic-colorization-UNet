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
        recreated_image= torchvision.transforms.functional.to_pil_image(recreated_image[0])
        st.image(recreated_image, caption="", use_column_width=True)


    # Display some other image on the other half of the page
    #display_other_image(output_image)



if __name__ == "__main__":
    main()

