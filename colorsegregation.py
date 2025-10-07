import cv2 as cv
import numpy as np

def contar_objetos(mascara):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mascara, connectivity=8)
    
    return num_labels - 1

img = cv.imread('figura.png')
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)    

rojo_bajo1 = np.array([0, 100, 80])
rojo_alto1 = np.array([10, 255, 255])
rojo_bajo2 = np.array([170, 100, 80])
rojo_alto2 = np.array([180, 255, 255])

verde_bajo = np.array([35, 100, 80])
verde_alto = np.array([85, 255, 255])

azul_bajo = np.array([100, 100, 80])
azul_alto = np.array([130, 255, 255])

amarillo_bajo = np.array([20, 100, 80])
amarillo_alto = np.array([30, 255, 255])

mascara_roja1 = cv.inRange(img_hsv, rojo_bajo1, rojo_alto1)
mascara_roja2 = cv.inRange(img_hsv, rojo_bajo2, rojo_alto2)
mascara_roja = cv.add(mascara_roja1, mascara_roja2)

mascara_verde = cv.inRange(img_hsv, verde_bajo, verde_alto)
mascara_azul = cv.inRange(img_hsv, azul_bajo, azul_alto)
mascara_amarillo = cv.inRange(img_hsv, amarillo_bajo, amarillo_alto)

rojos = contar_objetos(mascara_roja)
verdes = contar_objetos(mascara_verde)
azules = contar_objetos(mascara_azul)
amarillos = contar_objetos(mascara_amarillo)

print(f"rojos: {rojos}, verdes: {verdes}, azules: {azules}, amarillos: {amarillos}")

cv.imshow('rojos', mascara_roja)
cv.imshow('verdes', mascara_verde)
cv.imshow('azules', mascara_azul)
cv.imshow('amarillos', mascara_amarillo)

cv.waitKey(0)
cv.destroyAllWindows()