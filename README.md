# projkit

Welcome to **projkit**, a Python library designed to simplify camera projection tasks and calculations,
particularly when working with image predictions and 3D point cloud data. 
This library provides functions to effectively incorporate point cloud data with image predictions.


## Features

- **Camera Projection to Image Coordinates**: Easily project point cloud data onto image coordinates using provided camera parameters.
  
  ```python
  from projkit.camops import project_in_2d_with_K_R_t_dist_coeff
  from projkit.imutils import to_image, filter_image_and_world_points_with_img_dim

  ic, wc, z = project_in_2d_with_K_R_t_dist_coeff(K, R, t, d, wc)
  ic, wc, z = filter_image_and_world_points_with_img_dim(Nx, Ny, ic, wc)

  projection_on_image = to_image(Ny, Nx, ic, wc)
  ```
  
- **Intersection with Binary Mask**: Determine intersections between projected data and a binary mask.
  
  ```python
  from projkit.imutils import intersection
  binary_mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
  binary_mask[binary_mask > 0.50] = 255

  intersection_img, locations = intersection(binary_mask, ic, wc)
  ```
  
- **Identifying Data Holes in Mask**: Identify locations in the mask that require interpolation due to missing point cloud data.
  
  ```python
  import numpy as np
  from projkit.imutils import difference

  _missing_z_values_image = difference(Ny, Nx, ic, wc, binary_mask)
  x, y = np.where(_missing_z_values_image == 255)
  locations = list(zip(y, x))
  ```
  
- **Nearest Search Interpolation**: Perform nearest search interpolation for dense regions in point cloud data.
  
  ```python
  from projkit.imutils import nn_interpolation

  query = nn_interpolation(ic, wc)
  points = query.generate_points_for_nn_search(Ny, Nx, binary_mask)
  ic, wc, dist = query.query(points, dist_thresh=15)
  ```
  
  For larger datasets, utilize batch processing:
  
  ```python
  from projkit.imutils import nn_interpolation
  from projkit.pyutils import batch_gen

  query = nn_interpolation(ic, wc)
  points = query.generate_points_for_nn_search(Ny, Nx, binary_mask)
  for i, batch in batch_gen(points, batch_size=500):
      ic, wc, dist = query.query(batch, dist_thresh=15)
  ```

[comment]: <> (## Installation)

[comment]: <> (To use **projkit**, simply install it using pip:)

[comment]: <> (```sh)

[comment]: <> (pip install projkit)

[comment]: <> (```)

