import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carregando Imagens
Imagem = cv2.imread('C:/Users/Anderson/Desktop/processamentoDigitalDeImagens/diaretdb1/diaretdb1_image003.png')
Imagem2 = cv2.imread('C:/Users/Anderson/Desktop/processamentoDigitalDeImagens/diaretdb1/diaretdb1_image021.png')

# Diminuindo tamanho das Imagens
scalaPercent = 20
width = int(Imagem.shape[1] * scalaPercent / 100)
height = int(Imagem.shape[0] * scalaPercent / 100) 
dim = (width, height)
I1 = cv2.resize(Imagem, dim, interpolation=cv2.INTER_AREA)
I2 = cv2.resize(Imagem2, dim, interpolation=cv2.INTER_AREA)

# Exibindo Imagens
final = np.concatenate((I1, I2), axis=1)  # Concatenar horizontalmente
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.title("Figura 1 - Retina saudável e retina com retinopatia diabética")
plt.axis('off')
plt.show()
