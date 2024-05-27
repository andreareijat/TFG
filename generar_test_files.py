import os

def generate_test_files_txt(data_path, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):  # Asegúrate de filtrar solo archivos de imagen
                    relative_path = os.path.relpath(root, data_path)
                    file_name = os.path.splitext(file)[0]
                    # Escribe en el formato similar al de KITTI
                    f.write(f"{relative_path} {file_name} l\n")

data_path = './monodepth2/datasets/monocular_photos'
output_file = './monodepth2/splits/eigen/test_files2.txt'
generate_test_files_txt(data_path, output_file)
print(f"Archivo {output_file} generado correctamente.")
