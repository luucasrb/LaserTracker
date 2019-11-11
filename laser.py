import sys
import numpy as np
import cv2

#Escolhe o video que vai rodar

qual_video = 1

#Seta o video para captura de frames
if qual_video == 1:
    vid = cv2.VideoCapture('laser.mp4')
elif qual_video == 2:
    vid = cv2.VideoCapture('laser2.mp4')
elif qual_video == 3:
    vid = cv2.VideoCapture('laser3.3gp')
elif qual_video == 4:
    vid = cv2.VideoCapture('laser4.3gp')
elif qual_video == 5:
    vid = cv2.VideoCapture('laser5.3gp')

while(vid.isOpened()):
    #Le frame a frame do video
    success, frame = vid.read()
    
    #Se o video terminou: seta um erro no console
    if success == False:
        sys.stderr.write("Vídeo Terminou\n")
        cv2.waitKey(1)
        break
        
    #Limites de cor
    lower_red = np.array([0,0,255]) 
    upper_red = np.array([255,255,255]) 

    #Variavel usado para os operadores morfologico OPEN e CLOSE
    kernel = np.ones((3,3),np.uint8)

    #Thresholding usando inRange e Morph_OPEN como Máscara
    #inRange é uma funcao semelhante a funcao threshold, porem com um range de valores min e max
    
    mask_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(mask_HSV, lower_red, upper_red) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    output_mask = cv2.bitwise_and(frame, frame, mask = mask)
    
    #Faz o circulo vermelho em volta do laser
    (minVal, maxVal,minLoc, maxLoc) = cv2.minMaxLoc(mask) #Encontra o valor max na máscara
    cv2.circle(frame, maxLoc, 20, (0, 0, 255), 2, cv2.LINE_AA) #Desenha o circulo

    #Transformada de Hough
    output_hough = cv2.cvtColor(output_mask, cv2.COLOR_BGR2GRAY) 
    output_hough = cv2.medianBlur(output_hough, 5)
    circles = cv2.HoughCircles(output_hough, cv2.HOUGH_GRADIENT, 1, 20, param1=18, param2=8, minRadius=0, maxRadius=15)
 
    #Circulo verde em cima do laser 
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(frame,(i[0],i[1]),2,(0,255,0),3)
            
    #Exibe as três janelas
    nomeMask = "HSV Mask: inRange, MORPH_OPEN, MORPH_CLOSE e bitwise_and"
    nomeRGB = "RGB Frame: Cicle e HoughCircules desenhados"
    nomeHough = "GRAY Transformada de Hough: medianBlur"
    
    #Mostra a mascara em escala HSV
    cv2.namedWindow(nomeMask, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nomeMask, 1536 , 864)
    cv2.imshow(nomeMask, output_mask)
    cv2.moveWindow(nomeMask, 0, 0)
    
    #Mostra o frame com os circulos de Hough em escala RGB
    cv2.namedWindow(nomeRGB, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nomeRGB, 1536 , 864)
    cv2.imshow(nomeRGB, frame)
    cv2.moveWindow(nomeRGB, 1500, 1000)
    
    #Mostra a Transformacao de Hough em escala GRAY
    cv2.namedWindow(nomeHough, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nomeHough, 1536 , 864)
    cv2.imshow(nomeHough, output_hough)
    cv2.moveWindow(nomeHough, 300, 300)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
