import os

def rename_images(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter only PNG files
    png_files = [f for f in files if f.lower().endswith('.png')]
    
    # Sort files to ensure correct numbering
    png_files.sort()
    
    # Enumerate and rename files
    for index, filename in enumerate(png_files):
        new_name = f"{index + 1:06}.jpg"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

# Replace 'your_directory_path' with the path to your images directory
rename_images('./monocular_photos/')
