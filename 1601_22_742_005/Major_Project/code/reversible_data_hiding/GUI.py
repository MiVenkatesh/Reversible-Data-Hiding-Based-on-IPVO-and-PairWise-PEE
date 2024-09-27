import os
from unittest import result
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

from skimage import transform
from src.data_extraction import psnr_value, efv, eper_value,restore_image

import matplotlib.pyplot as plt
import shutil

from src.image_preprocessing import load_image, convert_to_grayscale
from src.adaptive_ipvo import generate_histogram, find_optimal_threshold
from src.two_segment_pairwise_pee import two_segment_pairwise_pee
from src.data_embedding import embed_data_in_block
from src.performance_evaluation import payload_capacity, embedding_efficiency, psnr, calculate_eper
#from src.gui import GUI

'''def save_image(image_path, image_data):
    img = Image.fromarray(image_data)
    img.save(image_path)
'''
def save_image(image_path, image_data):
    # If the image_data is a NumPy array, convert it to a PIL image
    if isinstance(image_data, np.ndarray):
        # Convert the NumPy array to a PIL image
        img = Image.fromarray(image_data)
    else:
        img = image_data

    # Convert the image to RGB mode if it's not already in RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save the image
    img.save(image_path)


def divide_image_into_blocks(image, block_width, block_height):
    height, width = image.shape
    blocks = []
    for y in range(0, height - height % block_height, block_height):
        for x in range(0, width - width % block_width, block_width):
            block = image[y:y+block_height, x:x+block_width]
            blocks.append(block)
    return blocks

def combine_blocks(blocks, block_width, block_height, grayscale_image):
    height, width = grayscale_image.shape
    num_rows = height // block_height
    num_cols = width // block_width

    rows = []
    for i in range(num_rows):
        row = np.concatenate(blocks[i*num_cols:(i+1)*num_cols], axis=1)
        rows.append(row)

    combined_image = np.concatenate(rows, axis=0)

    grayscale_image_resized = transform.resize(grayscale_image, (height, width))
    combined_image_resized = transform.resize(combined_image, (height, width))

    return combined_image_resized, grayscale_image_resized

def data_to_binary(data, width, height):
    binary_data = []
    total_pixels = (width - 1) * (height - 1)
    data_bytes = data.encode()
    for byte in data_bytes:
        bits = bin(byte)[2:].zfill(8)
        binary_data.extend([int(bit) for bit in bits])
    padding = total_pixels - len(binary_data)
    binary_data.extend([0] * padding)
    return binary_data

def main():
    # Create the GUI window
    root = tk.Tk()
    root.title("Reversible Data Hiding")
    root.state('zoomed')
   
    

    # Function to handle embedding when the Embed button is clicked
    def embed_data():
        global input_image_path
        image_path = input_image_path
        data = data_entry.get()
        output_dir = output_dir_entry.get()
        

        block_sizes = ["2x2", "3x3", "4x4", "5x5"]

        results = []
        image_files = [os.path.basename(image_path)]

        for image_file in image_files:
            output_image_path = os.path.join(output_dir, image_file)

            image = load_image(image_path)
            if image.ndim == 3 and image.shape[2] == 3:
                grayscale_image = convert_to_grayscale(image)
            else:
                grayscale_image = image
            histogram = generate_histogram(grayscale_image)

            threshold = find_optimal_threshold(histogram)
            optimized_histogram, two_segment_threshold = two_segment_pairwise_pee(grayscale_image)

            binary_data = data_to_binary(data, image.shape[1], image.shape[0])

            capacities = []
            psnrs = []

            for block_size in block_sizes:
                block_width, block_height = map(int, block_size.split("x"))

                blocks = divide_image_into_blocks(grayscale_image, block_width, block_height)
                embedded_images = []

                for block in blocks:
                    embedded_block = embed_data_in_block(block, optimized_histogram, two_segment_threshold, binary_data)
                    embedded_images.append(embedded_block)

                embedded_image, grayscale_image_resized = combine_blocks(embedded_images, block_width, block_height, grayscale_image)

                if block_size != "2x2":
                    embedded_image = np.clip(embedded_image, 0, 255).astype(np.uint8)

                save_image(os.path.join(output_dir, f"embedded_image_{block_size}_{image_file}"), embedded_image)

                payload_capacity_value = payload_capacity(embedded_image)
                embedding_efficiency_value = embedding_efficiency(embedded_image, grayscale_image)

                grayscale_image = grayscale_image.astype(np.float64)
                embedded_image = embedded_image.astype(np.float64)

                psnr_val = psnr(grayscale_image, embedded_image, data_range=grayscale_image.max() - grayscale_image.min())
                eper_ = calculate_eper(grayscale_image_resized, embedded_image)

                psnr_val = psnr_value(psnr_val)
                eper_val = eper_value(eper_, block_size)
                embedding_efficiency_value = efv(embedding_efficiency_value)
                result = {
                    "Image": image_file,
                    "Block Size": block_size,
                    "Payload Capacity": payload_capacity_value,
                    "Embedding Efficiency": embedding_efficiency_value,
                    "PSNR": psnr_val,
                    "EPER": eper_val,
                    "Output":data
                }
                #
                results.append(result)
                capacities.append(embedding_efficiency_value)
                psnrs.append(psnr_val)

        results_df = pd.DataFrame(results)
        averages = results_df.mean(numeric_only=True)
        averages["Image"] = "Average"
        averages["Block Size"] = "Average"
        results_df.loc[len(results_df)] = averages

        print(results_df)
        results_df.to_csv("results.csv", index=False)

        messagebox.showinfo("Success", "Embedding process completed.")
        # Restore the image
        #restored_image = restore_image(embedded_image, embedded_image)
        #print(restored_image)
        #save_image(os.path.join(output_dir, f"restored_{block_size}_{image_file}"), restored_image)

    # Function to handle selecting an image file
    def select_image():
        global input_image_path
        input_image_path = filedialog.askopenfilename(filetypes=[("All Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff")])
        #input_image_label.config(text=input_image_path)
        input_image_label_entry.config(text=input_image_path)


    label = tk.Label(root,text ='High-Fidelity Reversible Data Hiding Based on Improved Pixel Value Ordering and pairwise PEE')
    label.grid(row=0, column=2)

    # Create input image label and button
    input_image_label = tk.Label(root, text="Input Image:")
    input_image_label.grid(row=1, column=0)
    input_image_button = tk.Button(root, text="Browse", command=select_image)
    input_image_button.grid(row=1, column=1)
    input_image_label_entry = tk.Label(root, text="")
    input_image_label_entry.grid(row=1, column=2)

    # Create data entry label and entry
    data_label = tk.Label(root, text="Data to Embed:")
    data_label.grid(row=3, column=0)
    data_entry = tk.Entry(root, width=50)
    data_entry.grid(row=3, column=1)

     # Create output directory label and entry
    output_dir_label = tk.Label(root, text="Output Directory:")
    output_dir_label.grid(row=5, column=0)
    output_dir_entry = tk.Entry(root, width=75)
    output_dir_entry.grid(row=5, column=2)
    output_dir_button = tk.Button(root, text="Browse", command=lambda: output_dir_entry.insert(tk.END, filedialog.askdirectory()))
    output_dir_button.grid(row=5, column=1)

    # Create embed button
    embed_button = tk.Button(root, text="Embed", command=embed_data)
    embed_button.grid(row=8, column=1)

    def clear_inputs():
        # Clear input image path
        input_image_label_entry.config(text="")
        # Clear data entry
        data_entry.delete(0, tk.END)
        # Clear output directory entry
        output_dir_entry.delete(0, tk.END)

    # Create clear inputs button
    clear_inputs_button = tk.Button(root, text="Clear Inputs", command=clear_inputs)
    clear_inputs_button.grid(row=8, column=2)

    # Run the GUI
    root.mainloop()

if __name__ == "__main__":
   main()