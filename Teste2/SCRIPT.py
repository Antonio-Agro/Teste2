import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


def calcular_porcentagem_area_infestada(selected_contours, total_area):
    area_infestada = sum(cv2.contourArea(cnt) for cnt in selected_contours)
    porcentagem_area_infestada = (area_infestada / total_area) * 100
    return porcentagem_area_infestada


def display_image_with_text(image, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


df = pd.DataFrame(columns=['Imagem', 'Arquivo', 'Nivel', 'Numero de pulgoes', 'Constatacao'])


dir_path = 'C:\\Users\\anton\\Desktop\\Backup\\MESTRADO\\PROJETO\\SCRIPT\\IMAGENS\\15.07.24P\\GENOTIPO 3'

results = []  # Define a variável results como uma lista vazia


for filename in os.listdir(dir_path):
   
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
      
        filepath = os.path.join(dir_path, filename)
      
        img = cv2.imread(filepath)
        
      
        display_image_with_text(img, "Imagem Original")

      
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        display_image_with_text(gray, "Escala de Cinza")

       
        blur = cv2.GaussianBlur(gray, (5, 5), 5)
        display_image_with_text(blur, "Filtro de Suavização (Gaussian Blur)")

     
        ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        display_image_with_text(thresh, "Thresholding")

       
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

       
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

       
        ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

     
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(opening, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)

       
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        theta = lines[0][0][1]
        rho = lines[0][0][0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        
        nervura_pos = x1

       
        roi_width = 0.7 * 100  # 12 cm em pixels
        roi_height = img.shape[0]
        roi_x = nervura_pos - roi_width // 2
        roi_y = 0
        roi = (roi_x, roi_y, roi_width, roi_height)

        # Seleciona somente os contornos dentro da ROI e maiores que 0.4 mm
        min_area = 0.0004 / 25.4 * 300 ** 2  # Converte 0.4 mm para pixels considerando uma resolução de 300 DPI
        selected_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    distance = abs(cx - img.shape[1] // 2)
                if distance < 325:
                    selected_contours.append(cnt)

        
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

       
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

       
        mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

       
        pulgoes_highlighted = cv2.bitwise_and(img, img, mask=mask)

        
        pulgoes_highlighted_bgr = cv2.cvtColor(pulgoes_highlighted, cv2.COLOR_HSV2BGR)

        
        pulgoes_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        cv2.drawContours(pulgoes_highlighted_bgr, pulgoes_contours, -1, (0, 0, 255), 3)

       
        num_pulgões = len(selected_contours)
        margem_erro = num_pulgões * 0.15
        num_pulgões_min = round(num_pulgões - margem_erro)
        num_pulgões_max = round(num_pulgões + margem_erro)

        # Classificação com base no número de pulgões
        if num_pulgões == 0:
            nivel = "Nível 0: Ausência de pulgões"
            constatacao = "Folha saudável"
        elif num_pulgões <= 5:
            nivel = "Nível 1: Presença de até 5 pulgões"
            constatacao = "Folha com baixa infestação"
        elif num_pulgões <= 10:
            nivel = "Nível 2: Presença de 6 a 10 pulgões"
            constatacao = "Folha com baixa infestação"
        elif num_pulgões <= 20:
            nivel = "Nível 3: Presença de 11 a 20 pulgões"
            constatacao = "Folha com infestação moderada"
        elif num_pulgões <= 30:
            nivel = "Nível 4: Presença de 21 a 30 pulgões"
            constatacao = "Folha com infestação moderada"
        elif num_pulgões <= 40:
            nivel = "Nível 5: Presença de 31 a 40 pulgões"
            constatacao = "Folha com alta infestação"
        elif num_pulgões <= 50:
            nivel = "Nível 6: Presença de 41 a 50 pulgões"
            constatacao = "Folha com alta infestação"
        else:
            nivel = "Nível 7: Presença de mais de 50 pulgões"
            constatacao = "Folha com infestação severa"

        
        results.append([filename, nivel, num_pulgões, constatacao])


df = pd.DataFrame(results, columns=['Arquivo', 'Nível', 'Número de pulgoes', 'Constatacão'])

