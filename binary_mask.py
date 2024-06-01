from PIL import Image
import numpy as np
from eorision_and_dilation import binary_erosion, binary_dilation

# Cargar las imágenes
background = Image.open('Background.jpg').convert('L')
object_image = Image.open('Object.jpg').convert('L')

# Convertir las imágenes a matrices Numpy
background_array = np.array(background)
object_image_array = np.array(object_image)

# Restar el fondo de la imagen del objeto
difference = np.abs(object_image_array - background_array)  # Asegurarse de usar el valor absoluto

# Aplicar un umbral para obtener una máscara binaria
threshold = 30  # Ajusta este valor según sea necesario
binary_mask = np.where(difference > threshold, 1, 0).astype(np.uint8)

# Aplicar erosión y dilatación para eliminar falsos positivos y falsos negativos
binary_mask = binary_erosion(binary_mask)
binary_mask = binary_dilation(binary_mask)

# Convertir la máscara binaria a una imagen Pillow y guardarla
binary_mask_image = Image.fromarray((binary_mask * 255).astype('uint8'))
binary_mask_image.save('binary_mask.jpg')