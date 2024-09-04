import os
import h5py
import shutil

import numpy as np
import pandas as pd


def roi2array(path):
    coordinates = []
    with open(path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Split the line by whitespace
            line_data = line.split()
            # Check if the line has exactly 5 elements (index + x, y, z, and a value)
            if len(line_data) == 5:
                try:
                    # Convert the relevant columns to integers (assuming indices start at 0)
                    x = int(float(line_data[1]))
                    y = int(float(line_data[2]))
                    z = int(float(line_data[3]))
                    coordinates.append((z, x, y,))
                except ValueError:
                    # Skip the line if conversion fails
                    continue

    # Ensure we have valid coordinates
    if not coordinates:
        raise ValueError("No valid coordinates found in the file.")

    # Step 2: Determine the min and max for x, y, z
    min_z = min(coord[0] for coord in coordinates)
    max_z = max(coord[0] for coord in coordinates)
    min_x = min(coord[1] for coord in coordinates)
    max_x = max(coord[1] for coord in coordinates)
    min_y = min(coord[2] for coord in coordinates)
    max_y = max(coord[2] for coord in coordinates)

    # Step 3: Create a 3D array of zeros with dimensions adjusted by min values
    roi_3d_array = np.zeros((max_z - min_z + 1, max_x - min_x + 1, max_y - min_y + 1), dtype=int)

    # Step 4: Set the corresponding indices to 1, adjusted by min values
    for z, x, y in coordinates:
        roi_3d_array[z - min_z, x - min_x, y - min_y] = 1

    # Display the shape of the resulting 3D array and a summary
    return {"array":roi_3d_array,
            "cz":(min_z+max_z)/2,
            "cx":(min_x+max_x)/2,
            "cy":(min_y+max_y)/2,
            }

def create_h5(path, filename, outpath):
    # unzip zip file as temp
    shutil.unpack_archive(os.path.join(path, filename+".zip"), os.path.join(path, "temp"))
    # read csv
    df = pd.read_csv(os.path.join(path, filename+".csv"))
    metrix_names = df.columns.tolist()

    with h5py.File(os.path.join(outpath, filename)+".h5", "w") as f:
        for cell in os.listdir(os.path.join(path, "temp")):
            name  = (os.path.splitext(os.path.split(cell)[1])[0])
            items = roi2array(os.path.join(path, "temp", cell))
            f.create_dataset(
                name,
                data=items["array"])
            annotations = df[df["Name"]==name]
            for metrix_name in metrix_names:
                f[name].attrs[metrix_name] = annotations[metrix_name].item()
    # delete temp
    shutil.rmtree(os.path.join(path, "temp"))

if __name__ == "__main__":
    path = input("input your folder.\n"+\
                 " example: '_apply_JNet_617/C1/MD495_1G2_D14_FINC1'")
    outpath  = "_apply_JNet_617/microglia"
    filenames = sorted(set([os.path.splitext(i)[0] for i in os.listdir(path)]))
    for filename in filenames:
        create_h5(path, filename, outpath)