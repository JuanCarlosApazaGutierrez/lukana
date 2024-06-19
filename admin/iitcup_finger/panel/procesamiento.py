import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
imagen = cv2.imread('imagen_ejemplo.jpg', cv2.IMREAD_GRAYSCALE)

# Calcular los gradientes en direcciones x e y utilizando el operador Sobel
gradiente_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
gradiente_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)

# Calcular la magnitud del gradiente
magnitud_gradiente = np.sqrt(gradiente_x**2 + gradiente_y**2)

# Calcular la dirección del gradiente (en radianes)
direccion_gradiente = np.arctan2(gradiente_y, gradiente_x)

# Mostrar los resultados
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')

plt.subplot(1, 3, 2)
plt.imshow(magnitud_gradiente, cmap='gray')
plt.title('Magnitud del Gradiente')

plt.subplot(1, 3, 3)
plt.imshow(direccion_gradiente, cmap='hsv')
plt.title('Dirección del Gradiente')

plt.show()
