import os
from pdf2image import convert_from_path
def extract_images_from_pdfs_in_folder(folder_path, output_base_folder):
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    results = {}

    for pdf_file in pdf_files:
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_path = os.path.join(folder_path, pdf_file)
        pdf_output_folder = os.path.join(output_base_folder, pdf_name)

        if not os.path.exists(pdf_output_folder):
            os.makedirs(pdf_output_folder)

        images = convert_from_path(pdf_path,thread_count=os.cpu_count() - 1)
        image_paths = []

        for i, img in enumerate(images):
            image_path = os.path.join(pdf_output_folder, f"page_{i + 1}.png")
            img.save(image_path, "PNG")
            image_paths.append(image_path)
            print(f"saving {image_path}")

extract_images_from_pdfs_in_folder("./pdf-files","./pdf-images")