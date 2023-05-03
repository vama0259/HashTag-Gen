import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import itertools
from nltk.corpus import stopwords
import nltk
import easyocr
import numpy as np
nltk.download('stopwords')

# load the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
reader = easyocr.Reader(['en'])

# set up Streamlit app
st.set_page_config(layout='wide', page_title='Image Hashtag Recommender')


# define function to extract image features and generate hashtags
def generate_hashtags(image_file):
    # get image and convert to RGB mode
    image = Image.open(image_file).convert('RGB')

    # extract image features
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values)

    # decode the model output to text and extract caption words
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    caption_words = [word.lower() for word in output_text.split() if not word.startswith("#")]

    # remove stop words from caption words
    stop_words = set(stopwords.words('english'))
    caption_words = [word for word in caption_words if word not in stop_words]

    # use easyocr to extract text from the image
    text = reader.readtext(np.array(image))
    detected_text = " ".join([item[1] for item in text])

    # combine caption words and detected text
    all_words = caption_words + detected_text.split()

    # generate combinations of words for hashtags
    hashtags = []
    for n in range(1, 4):
        word_combinations = list(itertools.combinations(all_words, n))
        for combination in word_combinations:
            hashtag = "#" + "".join(combination)
            hashtags.append(hashtag)

    # return top 10 hashtags by frequency
    top_hashtags = [tag for tag in sorted(set(hashtags), key=hashtags.count, reverse=True) if tag != "#"]
    return top_hashtags[:10]


# display the Streamlit app
st.title("Image Hashtag Recommender")

image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if the user has submitted an image, generate hashtags
if image_file is not None:
    try:
        hashtags = generate_hashtags(image_file)
        if len(hashtags) > 0:
            st.write("Top 10 hashtags for this image:")
            for tag in hashtags:
                st.write(tag)
        else:
            st.write("No hashtags found for this image.")
    except Exception as e:
        st.write(f"Error: {e}")
