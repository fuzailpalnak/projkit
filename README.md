# projkit 

A Python library with utility functions for camera projection and computations on the projected data,
making it particularly useful when working with point cloud data and dealing with projection tasks.


### Usage

1. Given camera parameters, [project point cloud data to image cordinates](example/ex.ipynb#project_data)
```python
from projkit.camops import project_in_2d_with_K_R_t_dist_coeff
from projkit.imutils import to_image, filter_image_and_world_points_with_img_dim

ic, wc, z = project_in_2d_with_K_R_t_dist_coeff(K, R, t, d, wc)
ic, wc, z = filter_image_and_world_points_with_img_dim(Nx, Ny, ic, wc)

projectection_on_image = to_image(Ny, Nx, ic, wc)

```
2. [Find Intersection between projected data and binary mask](example/ex.ipynb#intersection)
```python
from projkit.imutils import intersection
binary_mask = cv2.imread(
                file, cv2.IMREAD_GRAYSCALE
            )
binary_mask[binary_mask > 0.50] = 255

intersection_img, locations = intersection(binary_mask, ic, wc)
```