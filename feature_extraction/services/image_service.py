try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract

BASIC_PATH = "../data/clickbait-training/"
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def calculate_image_features(entry):
    has_image = 0
    text_image = 0
    if len(entry["postMedia"]) > 0:
        has_image = 1
        text = get_text_from_image(entry)
        if text is not "":
            text_image = 1

    return has_image, text_image


def get_feat_names():
    return 'has image', 'text in image'


def get_text_from_image(entry):
    text = ""
    for img in entry["postMedia"]:
        text = pytesseract.image_to_string(Image.open(BASIC_PATH + img))
    return text
