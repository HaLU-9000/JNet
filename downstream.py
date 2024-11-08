import os
import h5py
import shutil

import numpy as np
import pandas as pd

from skimage import measure
import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace

from geomstats.geometry.discrete_curves import DiscreteCurvesStartingAtOrigin
from geomstats.learning.frechet_mean import FrechetMean


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


class CellShapeAnalysis():
    def __init__(self,k_sampling_points, dim):
        self.k_sampling_points = k_sampling_points
        self.dim = dim

        self.PRESHAPE_SPACE = PreShapeSpace(
            ambient_dim=self.dim, k_landmarks=k_sampling_points)
        self.PRESHAPE_SPACE.equip_with_group_action("rotations")
        self.PRESHAPE_SPACE.equip_with_quotient()

        self.CURVES_SPACE_SRV = DiscreteCurvesStartingAtOrigin(
            ambient_dim=self.dim,
            k_sampling_points=self.k_sampling_points)

    def array_to_curve(self, array):

        verts, _, _, _ = measure.marching_cubes(array, level=0.5)

        return verts

    def interpolate(self, curve, nb_points):
        """Interpolate a discrete curve with nb_points from a discrete curve.

        Returns
        -------
        interpolation : discrete curve with nb_points points
        """
        old_length = curve.shape[0]
        dim        = curve.shape[1]
        interpolation = gs.zeros((nb_points, dim))
        incr = old_length / nb_points
        pos = 0
        for i in range(nb_points):
            index = int(gs.floor(pos))
            interpolation[i] = curve[index] + (pos - index) * (
                curve[(index + 1) % old_length] - curve[index]
            )
            pos += incr
        return interpolation

    def preprocess(self,curve, tol=1e-10):
        """Preprocess curve to ensure that there are no consecutive duplicate points.

        Returns
        -------
        curve : discrete curve
        """
        dist = curve[1:] - curve[:-1]
        dist_norm = np.sqrt(np.sum(np.square(dist), axis=1))

        if np.any(dist_norm < tol):
            for i in range(len(curve) - 1):
                if np.sqrt(np.sum(np.square(curve[i + 1] - curve[i]), axis=0)) < tol:
                    curve[i + 1] = (curve[i] + curve[i + 2]) / 2

        return curve

    def exhaustive_align(self, curve, base_curve):
        """Align curve to base_curve to minimize the L² distance.

        Returns
        -------
        aligned_curve : discrete curve
        """
        nb_sampling = len(curve)
        distances = gs.zeros(nb_sampling)
        base_curve = gs.array(base_curve)
        for shift in range(nb_sampling):
            reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
            aligned = self.PRESHAPE_SPACE.fiber_bundle.align(
                point=gs.array(reparametrized), base_point=base_curve
            )
            distances[shift] = self.PRESHAPE_SPACE.embedding_space.metric.norm(
                gs.array(aligned) - gs.array(base_curve)
            )
        shift_min = gs.argmin(distances)
        reparametrized_min = [
            curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
        ]
        aligned_curve = self.PRESHAPE_SPACE.fiber_bundle.align(
            point=gs.array(reparametrized_min), base_point=base_curve
        )

        return aligned_curve

    def process(self, curve):

        curve = self.interpolate(curve, nb_points=self.k_sampling_points)
        curve = self.preprocess(curve)
        curve = self.PRESHAPE_SPACE.projection(curve)

        return curve

    def process_all(self, curves, func):

        curves_new = []
        for curve in curves:
          curve_new = func(curve)
          curves_new.append(curve_new)

        return curves_new

    def process_all_from_basis(self, curves, basis, func):

        curves_new = []
        for curve in curves:
          curve_new = func(curve, basis)
          curves_new.append(curve_new)
        return curves_new

    def scatter_3d(self, curve):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(curve[:, 0], curve[:, 1], curve[:, 2])

        return fig

    def estimate_mean(self, curves):

        mean = FrechetMean(self.CURVES_SPACE_SRV)
        cell_shapes_at_origin = self.CURVES_SPACE_SRV.projection(curves)
        mean.fit(cell_shapes_at_origin)

        return mean.estimate_

    def distance_from_mean(self, curve, mean_estimate):

        distance = self.CURVES_SPACE_SRV.metric.dist(
            self.CURVES_SPACE_SRV.projection(curve), mean_estimate
        )

        return distance

if __name__ == "__main__":
    path = input("input your folder.\n"+\
                 " example: '_apply_JNet_617/C1/MD495_1G2_D14_FINC1'")
    outpath  = "_apply_JNet_617/microglia"
    filenames = sorted(set([os.path.splitext(i)[0] for i in os.listdir(path)]))
    for filename in filenames:
        create_h5(path, filename, outpath)