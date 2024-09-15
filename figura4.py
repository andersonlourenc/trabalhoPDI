import cv2
import matplotlib.pyplot as plt
import numpy as np

# Caminho da Imagem
caminho_imagem = 'C:/Users/Anderson/Desktop/trabalhoPDI/diaretdb1/diaretdb1_image019.png'

# Carregar Imagem
imagem = cv2.imread(caminho_imagem)

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    raise FileNotFoundError(f"Não foi possível carregar a imagem do caminho: {caminho_imagem}")

# Diminuir tamanho da Imagem (ajuste o valor de dim conforme necessário)
scalaPercent = 20
width = int(imagem.shape[1] * scalaPercent / 100)
height = int(imagem.shape[0] * scalaPercent / 100)
dim = (width, height)
imagem = cv2.resize(imagem, dim, interpolation=cv2.INTER_AREA)

# Aplicar Filtro Gaussiano para eliminar ruído
imagem2 = cv2.GaussianBlur(src=imagem, ksize=(3, 3), sigmaX=0)

# Extração do canal verde
imagem3 = imagem.copy()
imagem3[:, :, 0] = 0  # Remover canal azul
imagem3[:, :, 2] = 0  # Remover canal vermelho

# Conversão para tons de cinza
imagem4 = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
imagem4 = cv2.cvtColor(imagem4, cv2.COLOR_GRAY2BGR)

# Concatenar imagens
final = np.concatenate((imagem, imagem2), axis=1)
final2 = np.concatenate((imagem3, imagem4), axis=1)
resultado_final = np.concatenate((final, final2), axis=0)

# Exibir Imagem Resultante
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(resultado_final, cv2.COLOR_BGR2RGB))
plt.title("Figura 4 - Etapas de pré-processamento - a)Imagem Original, b)Remoção de ruído, c)Extração do canal verde d)Conversão da Escala cinza")
plt.axis('off')
plt.show()
