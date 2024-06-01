from masks import sobel_x, sobel_y, log, sharpen
import numpy as np
from PIL import Image

def apply_convolution(image):
    conv_x = np.zeros_like(image)
    conv_y = np.zeros_like(image)
    conv_log = np.zeros_like(image)
    conv_sharpen = np.zeros_like(image)
    gradient_magnitude = np.zeros_like(image)
    
    # Apply convolution manually
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            conv_x[i, j] = (image[i-1, j-1] * sobel_x[0, 0] +
                            image[i-1, j] * sobel_x[0, 1] +
                            image[i-1, j+1] * sobel_x[0, 2] +
                            image[i, j-1] * sobel_x[1, 0] +
                            image[i, j] * sobel_x[1, 1] +
                            image[i, j+1] * sobel_x[1, 2] +
                            image[i+1, j-1] * sobel_x[2, 0] +
                            image[i+1, j] * sobel_x[2, 1] +
                            image[i+1, j+1] * sobel_x[2, 2])
            
            conv_y[i, j] = (image[i-1, j-1] * sobel_y[0, 0] +
                            image[i-1, j] * sobel_y[0, 1] +
                            image[i-1, j+1] * sobel_y[0, 2] +
                            image[i, j-1] * sobel_y[1, 0] +
                            image[i, j] * sobel_y[1, 1] +
                            image[i, j+1] * sobel_y[1, 2] +
                            image[i+1, j-1] * sobel_y[2, 0] +
                            image[i+1, j] * sobel_y[2, 1] +
                            image[i+1, j+1] * sobel_y[2, 2])
            
            conv_log[i, j] = (image[i-1, j-1] * log[0, 0] +
                             image[i-1, j] * log[0, 1] +
                             image[i-1, j+1] * log[0, 2] +
                             image[i, j-1] * log[1, 0] +
                             image[i, j] * log[1, 1] +
                             image[i, j+1] * log[1, 2] +
                             image[i+1, j-1] * log[2, 0] +
                             image[i+1, j] * log[2, 1] +
                             image[i+1, j+1] * log[2, 2])
            
            conv_sharpen[i, j] = (image[i-1, j-1] * sharpen[0, 0] +
                                  image[i-1, j] * sharpen[0, 1] +
                                  image[i-1, j+1] * sharpen[0, 2] +
                                  image[i, j-1] * sharpen[1, 0] +
                                  image[i, j] * sharpen[1, 1] +
                                  image[i, j+1] * sharpen[1, 2] +
                                  image[i+1, j-1] * sharpen[2, 0] +
                                  image[i+1, j] * sharpen[2, 1] +
                                  image[i+1, j+1] * sharpen[2, 2])
            
            gradient_magnitude[i, j] = np.sqrt(conv_x[i, j]**2 + conv_y[i, j]**2)
    
    return conv_x, conv_y, conv_log, conv_sharpen, gradient_magnitude

if __name__ == '__main__':
    # Load grayscale image
    grayscale_image = Image.open('Grayscale.jpg')
    grayscale_image = np.array(grayscale_image)

    # Load color image
    color_image = Image.open('Test.jpg')
    color_image = np.array(color_image)

    # Apply convolution to grayscale image
    conv_x_gray, conv_y_gray, conv_log_gray, conv_sharpen_gray, gradient_magnitude_gray = apply_convolution(grayscale_image)

    # Apply convolution to color image
    conv_x_color, conv_y_color, conv_log_color, conv_sharpen_color, gradient_magnitude_color = apply_convolution(color_image)

    # Convert the convolution results to images
    conv_x_gray_image = Image.fromarray(conv_x_gray)
    conv_y_gray_image = Image.fromarray(conv_y_gray)
    conv_log_gray_image = Image.fromarray(conv_log_gray)
    conv_sharpen_gray_image = Image.fromarray(conv_sharpen_gray)
    gradient_magnitude_gray_image = Image.fromarray(gradient_magnitude_gray)

    conv_x_color_image = Image.fromarray(conv_x_color)
    conv_y_color_image = Image.fromarray(conv_y_color)
    conv_log_color_image = Image.fromarray(conv_log_color)
    conv_sharpen_color_image = Image.fromarray(conv_sharpen_color)
    gradient_magnitude_color_image = Image.fromarray(gradient_magnitude_color)

    # Save the images
    conv_x_gray_image.save('conv_x_gray.jpg')
    conv_y_gray_image.save('conv_y_gray.jpg')
    conv_log_gray_image.save('conv_log_gray.jpg')
    conv_sharpen_gray_image.save('conv_sharpen_gray.jpg')
    gradient_magnitude_gray_image.save('gradient_magnitude_gray.jpg')

    conv_x_color_image.save('conv_x_color.jpg')
    conv_y_color_image.save('conv_y_color.jpg')
    conv_log_color_image.save('conv_log_color.jpg')
    conv_sharpen_color_image.save('conv_sharpen_color.jpg')
    gradient_magnitude_color_image.save('gradient_magnitude_color.jpg')