import numpy as np
from matplotlib import pyplot as plt
import cv2 #Opencv
import math
import skimage
from skimage import io

def ecualizacion_c (imagen_e):
    ecua_img = cv2.cvtColor(imagen_e,cv2.COLOR_BGR2YUV)
    ecua_img[:,:,0] = cv2.equalizeHist(ecua_img[:,:,0])
    res_ecua_img = cv2.cvtColor(ecua_img,cv2.COLOR_YUV2BGR)
    return res_ecua_img

def histograma_c (imagen_h):
    plt.figure(figsize=(4, 3))
    for i, c in enumerate(color): 
        hist_img = cv2.calcHist([imagen_h], [i], None, [256], [0, 256])
        plt.plot(hist_img, color = c)
        plt.xlim([0,256])
        
    plt.show()

def ecualizacion_g (imagen_e_g):
    ecua_img_g = cv2.equalizeHist(imagen_e_g)
    return ecua_img_g

def histograma_g (imagen_h_g):
    plt.figure(figsize=(4, 3))
    hist_img_g = cv2.calcHist([imagen_h_g], [0], None, [256], [0, 256])
    plt.plot(hist_img_g)
        
    plt.show()

#LECTURA IMAGENES{
img_1 = cv2.imread('m_1.jpg')
img_2 = cv2.imread('m_2.jpg')
#}

color = ('r','g','b')

#CAMBIAR TAMAÑO{
rimg_1 = cv2.resize(img_1, (300,300))
rimg_2 = cv2.resize(img_2, (300,300))
#}

#--------------------- INICIO IMAGEN 1 ---------------------------------#

e_img_1 = ecualizacion_c(rimg_1)#ECUALIZACION IMAGEN 1

cv2.imshow('IMAGEN 1',rimg_1) # SE MUESTRA IMAGEN 1
cv2.imshow('Ecualizacion Imagen 1',e_img_1)

h_img_1 = histograma_c(rimg_1)#HISTOGRAMA IMAGEN 1
h_e_img_1 = histograma_c(e_img_1)#HISTOGRAMA ECUALIZACION IMAGEN 1

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Ecualizacion Imagen 1') #SE CIERRA ECUALIZACION IMAGEN 1

#-------------------------------FIN IMAGEN 1-----------------------------------#

#--------------------- INICIO IMAGEN 2 ---------------------------------#

e_img_2 = ecualizacion_c(rimg_2)#ECUALIZACION IMAGEN 2

cv2.imshow('IMAGEN 2',rimg_2) # SE MUESTRA IMAGEN 2
cv2.imshow('Ecualizacion Imagen 2',e_img_2)

h_img_2 = histograma_c(rimg_2)#HISTOGRAMA IMAGEN 2
h_e_img_2 = histograma_c(e_img_2)#HISTOGRAMA ECUALIZACION IMAGEN 2

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Ecualizacion Imagen 2') #SE CIERRA ECUALIZACION IMAGEN 2

#-------------------------------FIN IMAGEN 2-----------------------------------#

#------------------------------- INICIO SUMA ----------------------------------#

suma = cv2.add(rimg_1,rimg_2)
e_suma = ecualizacion_c(suma)#ECUALIZACION SUMA

cv2.imshow('Suma',suma) #MUESTRA SUMA
cv2.imshow('Ecualizacion Suma',e_suma)

h_suma = histograma_c(suma)#HISTOGRAMA SUMA
h_e_suma = histograma_c(e_suma)#HISTOGRAMA ECUALIZACION SUMA

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Suma')
cv2.destroyWindow('Ecualizacion Suma')

#--------------------------------- FIN SUMA -----------------------------------#

#------------------------------- INICIO RESTA ----------------------------------#

resta = cv2.subtract(rimg_1,rimg_2)
e_resta = ecualizacion_c(resta)#ECUALIZACION RESTA

cv2.imshow('Resta',resta) #MUESTRA RESTA
cv2.imshow('Ecualizacion Resta',e_resta)

h_resta = histograma_c(resta)#HISTOGRAMA RESTA
h_e_resta = histograma_c(e_resta)#HISTOGRAMA ECUALIZACION RESTA

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Resta')
cv2.destroyWindow('Ecualizacion Resta')

#-------------------------------- FIN RESTA ------------------------------------#

#------------------------- INICIO MULTIPLICACION -------------------------------#

mult = cv2.multiply(rimg_1,rimg_2)
e_mult = ecualizacion_c(mult)#ECUALIZACION MULTIPLICACION

cv2.imshow('Multiplicacion',mult) #MUESTRA MULTIPLICACION
cv2.imshow('Ecualizacion Multiplicacion',e_mult)

h_mult = histograma_c(mult)#HISTOGRAMA MULTIPLICACION
h_e_mult = histograma_c(e_mult)#HISTOGRAMA ECUALIZACION MULTIPLICACION

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Multiplicacion')
cv2.destroyWindow('Ecualizacion Multiplicacion')

#--------------------------- FIN MULTIPLICACION --------------------------------#

#----------------------------- INICIO DIVISION ---------------------------------#

divi = cv2.divide(rimg_1,rimg_2)
e_divi = ecualizacion_c(divi)#ECUALIZACION DIVISION

cv2.imshow('Division',divi) #MUESTRA DIVISION
cv2.imshow('Ecualizacion Division',e_divi)

h_divi = histograma_c(divi)#HISTOGRAMA DIVISION
h_e_divi = histograma_c(e_divi)#HISTOGRAMA ECUALIZACION DIVISION

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Division')
cv2.destroyWindow('Ecualizacion Division')

#------------------------------- FIN DIVISION ----------------------------------#

#--------------------------------- INICIO RAIZ ---------------------------------#

raiz = (rimg_2**(0.5))
raiz_m = np.float32(raiz)
cv2.imwrite('raiz.png',raiz)
raiz_g = cv2.imread('raiz.png',0)
e_raiz = ecualizacion_g(raiz_g)#ECUALIZACION RAIZ

cv2.imshow('Raiz',raiz) #MUESTRA RAIZ
cv2.imshow('Ecualizacion Raiz',e_raiz)

h_raiz = histograma_c(raiz_m)#HISTOGRAMA RAIZ
h_e_raiz = histograma_g(e_raiz)#HISTOGRAMA ECUALIZACION RAIZ

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Raiz')
cv2.destroyWindow('Ecualizacion Raiz')

#---------------------------------- FIN RAIZ -----------------------------------#

#-------------------------------- INICIO DERIVADA ------------------------------#

derivada = cv2.Laplacian(rimg_1,cv2.CV_32F)
cv2.imwrite('der.png',derivada)
derivada_g = cv2.imread('der.png',0)
e_derivada = ecualizacion_g(derivada_g)#ECUALIZACION RAIZ


cv2.imshow('Derivada',derivada) #MUESTRA DERIVADA
cv2.imshow('Ecualizacion Derivada',e_derivada)

h_derivada = histograma_c(derivada)#HISTOGRAMA DERIVADA
h_e_derivada = histograma_g(e_derivada)#HISTOGRAMA ECUALIZACION DERIVADA

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Derivada')
cv2.destroyWindow('Ecualizacion Derivada')

#---------------------------------- FIN DERIVADA -------------------------------#

#-------------------------------- INICIO POTENCIA ------------------------------#

pot = cv2.pow(rimg_1,2)
e_pot = ecualizacion_c(pot)#ECUALIZACION POTENCIA

cv2.imshow('Potencia',pot) #MUESTRA POTENCIA
cv2.imshow('Ecualizacion Potencia',e_pot)

h_pot = histograma_c(pot)#HISTOGRAMA POTENCIA
h_e_pot = histograma_c(e_pot)#HISTOGRAMA ECUALIZACION POTENCIA

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Potencia')
cv2.destroyWindow('Ecualizacion Potencia')

#---------------------------------- FIN POTENCIA -------------------------------#

#-------------------------------- INICIO LOGARITMO -----------------------------#

logaritmo = np.zeros(rimg_1.shape, rimg_1.dtype)
c = 1
logaritmo = c * np.log(1+rimg_1)
maxi = np.amax(logaritmo)
logaritmo = np.uint8(logaritmo / maxi *255)
e_logaritmo = ecualizacion_c(logaritmo)#ECUALIZACION LOGARITMO

cv2.imshow('Logaritmo',logaritmo) #MUESTRA LOGARITMO
cv2.imshow('Ecualizacion Logaritmo',e_logaritmo)

h_logaritmo = histograma_c(logaritmo)#HISTOGRAMA LOGARTIMO
h_e_logaritmo = histograma_c(e_logaritmo)#HISTOGRAMA ECUALIZACION LOGARITMO

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Logaritmo')
cv2.destroyWindow('Ecualizacion Logaritmo')

#---------------------------------- FIN LOGARITMO ------------------------------#

#-------------------------------- INICIO CONJUNCION -----------------------------#

conjuncion = cv2.bitwise_and(rimg_1,rimg_2)
e_conjuncion = ecualizacion_c(conjuncion)#ECUALIZACION CONJUNCION

cv2.imshow('Conjuncion',conjuncion) #MUESTRA CONJUNCION
cv2.imshow('Ecualizacion Conjuncion',e_conjuncion)

h_conjuncion = histograma_c(conjuncion)#HISTOGRAMA CONJUNCION
h_e_conjuncion = histograma_c(e_conjuncion)#HISTOGRAMA ECUALIZACION CONJUNCION

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Conjuncion')
cv2.destroyWindow('Ecualizacion Conjuncion')

#---------------------------------- FIN CONJUNCION ------------------------------#

#-------------------------------- INICIO DISYUNCION -----------------------------#

disyuncion = cv2.bitwise_or(rimg_1,rimg_2)
e_disyuncion = ecualizacion_c(disyuncion)#ECUALIZACION DISYUNCION

cv2.imshow('Disyuncion',disyuncion) #MUESTRA DISYUNCION
cv2.imshow('Ecualizacion Disyuncion',e_disyuncion)

h_disyuncion = histograma_c(disyuncion)#HISTOGRAMA DISYUNCION
h_e_disyuncion = histograma_c(e_disyuncion)#HISTOGRAMA ECUALIZACION DISYUNCION

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Disyuncion')
cv2.destroyWindow('Ecualizacion Disyuncion')

#---------------------------------- FIN DIYUNCION -------------------------------#

#--------------------------------- INICIO NEGACION ------------------------------#

negacion = 1 - rimg_2
e_negacion = ecualizacion_c(negacion)#ECUALIZACION NEGACION

cv2.imshow('Negacion',negacion) #MUESTRA NEGACION
cv2.imshow('Ecualizacion Negacion',e_negacion)

h_negacion = histograma_c(negacion)#HISTOGRAMA NEGACION
h_e_negacion = histograma_c(e_negacion)#HISTOGRAMA ECUALIZACION NEGACION

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Negacion')
cv2.destroyWindow('Ecualizacion Negacion')

#----------------------------------- FIN NEGACION -------------------------------#

#--------------------------------- INICIO TRASLADO ------------------------------#

rows, cols, ch = rimg_1.shape
pts1 = np.float32([[50, 50],
                    [200, 50], 
                    [50, 200]])
pts2 = np.float32([[10, 100],
                    [200, 50], 
                    [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
tras = cv2.warpAffine(rimg_1, M, (cols, rows))
e_tras = ecualizacion_c(tras)#ECUALIZACION TRASLADO

cv2.imshow('Traslacion',tras) #MUESTRA TRASLADO
cv2.imshow('Ecualizacion Traslacion',e_tras)

h_tras = histograma_c(tras)#HISTOGRAMA TRASLADO
h_e_tras = histograma_c(e_tras)#HISTOGRAMA ECUALIZACION TRASLADO

cv2.waitKey(0) #Retardo
cv2.destroyWindow('Traslacion')
cv2.destroyWindow('Ecualizacion Traslacion')

#----------------------------------- FIN TRASLADO -------------------------------#

cv2.waitKey(0) #Retardo
cv2.destroyAllWindows()
