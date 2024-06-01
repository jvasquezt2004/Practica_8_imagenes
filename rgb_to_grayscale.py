import numpy as np
from PIL import Image

image_path = 'Test.jpg'
image = Image.open(image_path)

image_array = np.array(image)

def rgb_to_grayscale(image_array):
    height, width, _ = image_array.shape
    grayscale_array = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            r, g, b = image_array[i, j]
            grayscale_value = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
            grayscale_array[i, j] = grayscale_value
    return grayscale_array

if __name__ == '__main__':
    grayscale_image = rgb_to_grayscale(image_array)
    grayscale_image = Image.fromarray(grayscale_image)
    grayscale_image.save('Grayscale.jpg')