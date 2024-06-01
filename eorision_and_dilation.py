import numpy as np

def binary_erosion(image):
    # Convertir la imagen a un array de numpy de tipo uint8
    image_array = np.array(image, dtype=np.uint8)

    # Definir el elemento estructurante para la erosión
    structuring_element = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]], dtype=np.uint8)

    # Inicializar la imagen erosionada
    eroded_image = np.zeros_like(image_array)
    
    # Realizar la erosión binaria
    for i in range(1, image_array.shape[0] - 1):
        for j in range(1, image_array.shape[1] - 1):
            if np.all(image_array[i-1:i+2, j-1:j+2] & structuring_element == structuring_element):
                eroded_image[i, j] = 1

    return eroded_image

def binary_dilation(image):
    # Convertir la imagen a un array de numpy de tipo uint8
    image_array = np.array(image, dtype=np.uint8)

    # Definir el elemento estructurante para la dilatación
    structuring_element = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]], dtype=np.uint8)

    # Inicializar la imagen dilatada
    dilated_image = np.zeros_like(image_array)
    
    # Realizar la dilatación binaria
    for i in range(1, image_array.shape[0] - 1):
        for j in range(1, image_array.shape[1] - 1):
            if image_array[i, j] == 1:
                dilated_image[i-1:i+2, j-1:j+2] = np.bitwise_or(dilated_image[i-1:i+2, j-1:j+2], structuring_element)

    return dilated_image

if __name__ == '__main__':
    from PIL import Image
    
    # Cargar la imagen binaria
    image_path = '/path/to/binary/image.png'
    image = Image.open(image_path).convert('1')

    # Realizar la erosión binaria
    eroded_image = binary_erosion(image)

    # Realizar la dilatación binaria
    dilated_image = binary_dilation(image)

    # Convertir los arrays a imágenes
    eroded_image = Image.fromarray((eroded_image * 255).astype('uint8'))
    dilated_image = Image.fromarray((dilated_image * 255).astype('uint8'))

    # Guardar las imágenes erosionadas y dilatadas
    eroded_image.save('/path/to/eroded/image.png')
    dilated_image.save('/path/to/dilated/image.png')
