import os
import numpy as np

def convert_npy_folder(folder_path, output_folder=None, fmt="%.6f", delimiter=" "):
    """
    Convert all .npy files in folder_path to .txt files using numpy.savetxt.

    Args:
        folder_path (str): Directory containing .npy files.
        output_folder (str): Where to save .txt files (defaults to folder_path).  
        fmt (str): Format specifier for floats.  
        delimiter (str): Delimiter between values in text files.
    """
    output_folder = output_folder or folder_path
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if not filename.endswith(".npy"):
            continue
        npy_path = os.path.join(folder_path, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)

        print(f"Converting {npy_path} → {txt_path}")
        data = np.load(npy_path)
        
        # If array is >2D, flatten it or squeeze dimensions
        if data.ndim > 2:
            print(f"  - detected {data.ndim}D array, flattening to 2D")
            data = data.reshape(data.shape[0], -1)
        elif data.ndim == 3 and data.shape[1] == 1:
            # Common case: shape like (N, 1, M)
            print("  - squeezing extra dimension")
            data = np.squeeze(data)

        np.savetxt(txt_path, data, fmt=fmt, delimiter=delimiter)
    print("All conversions completed!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch convert .npy files to .txt.")
    parser.add_argument("input_folder", help="Folder containing .npy files")
    parser.add_argument("--output_folder", default=None, help="Folder to save .txt files")
    parser.add_argument("--fmt", default="%.6f", help="Numeric format (NumPy fmt)")
    parser.add_argument("--delimiter", default=" ", help="Delimiter in text files")
    args = parser.parse_args()

    convert_npy_folder(args.input_folder, args.output_folder, args.fmt, args.delimiter)
