# projkit 

A Python library with utility functions for camera projection and computations on the projected data,
making it particularly useful when working with point cloud data and dealing with projection tasks.


### Usage

1. Given camera parameters, [project point cloud data to image cordinates](example/project_point_cloud_to_image.ipynb)
```python
from projkit.camops import project_in_2d_with_K_R_t_dist_coeff
from projkit.imutils import to_image, filter_image_and_world_points_with_img_dim

ic, wc, z = project_in_2d_with_K_R_t_dist_coeff(K, R, t, d, wc)
ic, wc, z = filter_image_and_world_points_with_img_dim(Nx, Ny, ic, wc)

projectection_on_image = to_image(Ny, Nx, ic, wc)

```