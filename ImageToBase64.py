import base64
import cv2
# Read image file
with open("./Images/Julian.jpg", "rb") as image_file:
    image_data = image_file.read()
    encoded_string = base64.b64encode(image_data)

# Write base64 string to file
with open("./prueba.txt", "w") as output_file:
    output_file.write(encoded_string.decode('utf-8'))
