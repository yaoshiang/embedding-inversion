from PIL import Image
import pytesseract

# Load the image from which text needs to be extracted
image = Image.open('example.png')
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
print(data)