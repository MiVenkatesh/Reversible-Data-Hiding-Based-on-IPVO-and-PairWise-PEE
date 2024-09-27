# Reversible-Data-Hiding-Based-on-IPVO-and-PairWise-PEE

This project implements Reversible Data Hiding using Adaptive IPVO and Two-Segment Pairwise PEE. The aim is to embed user-provided data into an image in a reversible way, allowing the data to be extracted and the image to be restored with minimal distortion.

## Prerequisites

- Python 3.6 or later
- Compatible with Windows, macOS, and Linux operating systems

## Installation

1. Clone this repository:

   ```
   [git clone https://github.com/MiVenkatesh/](https://github.com/MiVenkatesh/Reversible-Data-Hiding-Based-on-IPVO-and-PairWise-PEE.git)
   ```

2. Change to the `reversible_data_hiding` directory:

   ```
   cd reversible_data_hiding
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Running 

1. Prepare an image to be used for data embedding. The image should be in a format supported by the Python Imaging Library (PIL), such as JPEG, PNG, or BMP. It is recommended to use lossless formats like PNG for better results.

2. Choose the data to be embedded. You can provide the data as a string using the `--data` command-line argument.

3. Run the project using the Command Line Interface (CLI):
   
4. Launch the GUI file (GUI.py).
   
5. Click "Select Input File" and choose an image (JPEG, PNG, JPG, TIFF, GIF).

6. Enter the data to be embedded in the "Enter Data" text box.

7. Click "Select Output Folder" and choose the desired output directory.

8. Click the "Embed" button to start the embedding process.

9. Wait for the pop-up message "Embedding process completed."

10. Check the PSNR (Peak Signal-to-Noise Ratio) value displayed in the terminal. A higher PSNR value indicates better image quality after the embedding and extraction process.

## Contributing Guidelines

1. Fork the repository on GitHub.
2. Clone your fork and create a new branch for your feature or bugfix.
3. Commit your changes to your branch, following the project's coding standards and adding appropriate test cases.
4. Push your changes to your fork on GitHub.
5. Submit a pull request to the main repository for review.

Please ensure that your code is well-documented, follows best practices, and passes all tests before submitting a pull request.
