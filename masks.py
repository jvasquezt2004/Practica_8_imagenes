import numpy as np

# Máscara de SobelX
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# Máscara de SobelY
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Máscara de LoG (Laplacian of Gaussian)
log = np.array([[0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]])

                # Máscara de perfilado
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]]) * 0.6