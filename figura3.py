import cv2
import matplotlib.pyplot as plt

# Abrir Imagem
imagem = cv2.imread('C:/Users/Anderson/Desktop/trabalhoPDI/imagemCorte/img1.jpg')

# Diminuir tamanho da Imagem
scalaPercent = 40
width = int(imagem.shape[1] * scalaPercent / 100)
height = int(imagem.shape[0] * scalaPercent / 100)
dim = (width, height)
I1 = cv2.resize(imagem, dim, interpolation=cv2.INTER_AREA)

# Aplicar Filtro Gaussiano
imagem2 = cv2.GaussianBlur(src=I1, ksize=(13, 13), sigmaX=0)

# Concatenar Imagens Horizontalmente
final = cv2.hconcat([I1, imagem2])

# Exibir Imagem Final
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.title("Figura 3 - Filtro Gaussiano - Imagem original e eliminação do ruído")
plt.axis('off')
plt.show()
