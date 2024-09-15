import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Caminho para a imagem (Atualize com o caminho local no seu computador)
image_path = 'C:/Users/Anderson/Desktop/trabalhoPDI/diaretdb1/diaretdb1_image007.png'

# Verificar se o arquivo existe
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"O arquivo não foi encontrado no caminho: {image_path}")
else:
    print(f"Arquivo encontrado: {image_path}")

# Carregar a imagem
Imagem4 = cv2.imread(image_path)

# Verificar se a imagem foi carregada
if Imagem4 is None:
    raise ValueError(f"Não foi possível carregar a imagem do caminho: {image_path}")
else:
    print(f"A imagem foi carregada com sucesso. Dimensões: {Imagem4.shape}")

# Exibir a imagem original para garantir que foi carregada corretamente
cv2.imshow('Imagem Original', Imagem4)
cv2.waitKey(0)  # Manter a janela aberta até pressionar uma tecla
cv2.destroyAllWindows()

# Convertendo para tons de cinza
Imagem4_gray = cv2.cvtColor(Imagem4, cv2.COLOR_BGR2GRAY)

# Aplicando Binarização
ret, thresh1 = cv2.threshold(Imagem4_gray, 127, 255, cv2.THRESH_BINARY)

# Eliminando falsos positivos com operação morfológica
image = thresh1.copy()
kernel = np.ones((5, 5), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Criando Imagem Binária
_, binary = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY_INV)

# Encontrando contornos
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Verificar se foram encontrados contornos
if contours:
    print(f"Número de contornos encontrados: {len(contours)}")
    
    # Identificar a região do disco óptico (OD) - Assumindo que o OD é o maior contorno
    od_contour = max(contours, key=cv2.contourArea)  # Selecionar o maior contorno como OD
    od_mask = np.zeros_like(Imagem4_gray)  # Cria uma máscara com o mesmo tamanho da imagem original em tons de cinza
    cv2.drawContours(od_mask, [od_contour], -1, 255, -1)  # Preenche o OD com branco na máscara

    # Inverter a máscara para remover o OD (branco no lugar do OD, preto no restante)
    od_mask_inv = cv2.bitwise_not(od_mask)

    # Remover OD da imagem binarizada (aplica a máscara invertida para manter tudo, exceto o OD)
    masked = cv2.bitwise_and(Imagem4_gray, Imagem4_gray, mask=od_mask_inv)

    # Mostrar o resultado usando matplotlib
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(thresh1, cmap='gray')
    plt.title('Imagem Binarizada')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(masked, cmap='gray')
    plt.title('Após Remoção do OD')
    plt.axis('off')

    plt.show()

else:
    print("Nenhum contorno encontrado. Verifique a imagem e os parâmetros de binarização.")
