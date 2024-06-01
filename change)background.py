from PIL import Image
import numpy as np

# Cargar las imágenes
object_image = Image.open('Object.jpg')
new_background = Image.open('New_Background.jpg')
mask = Image.open('binary_mask.jpg').convert('L')

# Redimensionar el fondo y la máscara para que coincidan con el tamaño de la imagen del objeto
new_background = new_background.resize(object_image.size)
mask = mask.resize(object_image.size)

# Convertir las imágenes a arrays de NumPy
object_array = np.array(object_image)
background_array = np.array(new_background)
mask_array = np.array(mask)

# Normalizar la máscara
mask_array = mask_array / 255
mask_inv_array = 1 - mask_array

# Aplicar la máscara al objeto y al nuevo fondo
object_image_masked_array = (object_array * mask_array[:, :, np.newaxis]).astype(np.uint8)
new_background_masked_array = (background_array * mask_inv_array[:, :, np.newaxis]).astype(np.uint8)

# Combinar las imágenes enmascaradas
result_array = object_image_masked_array + new_background_masked_array

# Convertir el array de resultado de vuelta a una imagen
result_image = Image.fromarray(result_array)

# Guardar la imagen resultante
result_image.save('result_image.jpg')
result_image.show()
