import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

# Directorios para guardar las imágenes
monocular_dir = "monocular_photos"
rgbd_dir = "rgbd_photos"

if not os.path.exists(monocular_dir):
    os.makedirs(monocular_dir)

if not os.path.exists(rgbd_dir):
    os.makedirs(rgbd_dir)

# Configurar el pipeline de RealSense
pipeline = rs.pipeline()
config = rs.config()

# Configuramos la cámara para capturar tanto color como profundidad
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Iniciar el pipeline
pipeline.start(config)

try:
    while True:
        # Esperar a que un conjunto de cuadros esté listo
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convertir imágenes a arrays de numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Muestra las imágenes en ventanas separadas
        cv2.imshow('Stream Monocular', color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Stream RGBD', np.hstack((color_image, depth_colormap)))

        key = cv2.waitKey(1)
        if key & 0xFF == ord(' '):  # Presiona la barra de espacio para capturar
            # Guardar la imagen de color (foto monocular)
            color_img_path = os.path.join(monocular_dir, f"color-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg")
            cv2.imwrite(color_img_path, color_image)

            # Guardar la imagen de profundidad
            depth_img_path = os.path.join(rgbd_dir, f"depth-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg")
            cv2.imwrite(depth_img_path, depth_colormap)

        elif key & 0xFF == ord('q'):  # Presiona 'q' para salir
            break
finally:
    # Detener y cerrar todo correctamente
    pipeline.stop()
    cv2.destroyAllWindows()
