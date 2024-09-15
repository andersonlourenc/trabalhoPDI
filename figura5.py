import cv2
import matplotlib.pyplot as plt
import numpy as np

# Caminho da Imagem (ajuste para o caminho correto no seu sistema)
caminho_imagem = 'C:/Users/Anderson/Desktop/trabalhoPDI/diaretdb1/diaretdb1_image007.png'

# Carregar Imagem
imagem = cv2.imread(caminho_imagem)

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    raise FileNotFoundError(f"Não foi possível carregar a imagem do caminho: {caminho_imagem}")

# Conversão para tons de cinza
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Gerar Histograma
plt.figure(figsize=(10, 6))
plt.hist(imagem_gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)

# Configurações do eixo x
plt.xlim([0, 270])
plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255])

# Configurações do eixo y
plt.ylim([0, 25000])
plt.yticks([0, 5000, 10000, 15000, 20000])

# Títulos e rótulos
plt.title("Histograma da Imagem em Tons de Cinza")
plt.xlabel('Intensidade de Pixels')
plt.ylabel('Frequência')

# Exibir o histograma
plt.grid(True)
plt.show()
