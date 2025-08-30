# Evaluation script

This section is designed to prepare the submission file for our [challenge](https://challenge.shannon.humansensing.cs.cmu.edu/web/challenges/challenge-page/21/overview). The script expects a very specific directory structure, and has certain default arguments that must not be modified to keep the scoring accurate.


## Expected directory structure

```bash
.
├── render_gaussians.py     # Main rendering script
├── cameras_path            # Camera parameters
|   ├── vr                  # Phase name
|   |   ├── <scene_name>.json
|   |   ├── ...
|   |   └── <8 camera files>
|   └── interp
|       ├── <scene_name>.json
|       ├── ...
|       └── <8 camera files>
└── pointcloud_path         # Input PLY files
    ├── <scene_name>
    |    └── point_cloud/
    |        ├── gaussians_0000.ply
    |        ├── gaussians_0001.ply
    |        └── ...
    ├── ...
    └── <8 scenes>
```

## Usage

### Prerequisites

```bash
conda env create -f environment.yml
conda activate basket
pip install git+https://github.com/Awesome3DGS/libgs.git

# Optional but highly recommended for better quality
# Install ffmpeg: https://ffmpeg.org/download.html
```

### Running on a scene

```bash
python render_gaussians.py \
        --pointcloud_path   <pointcloud_path> \
        --scene_name    <scene_name>    \
        --camera_file  <camera_json_file>
```

### Submission structure
We strongly recommend participants to organize the submission folder in the following structure before creating a zip. For more details, follow the instructions in our [competition page](https://challenge.shannon.humansensing.cs.cmu.edu/web/challenges/challenge-page/21/phases).

```bash
└── predictions
       ├── atc_1.mp4
       ├── atc_2.mp4
       ├── atc_3.mp4
       ├── atc_4.mp4
       ├── def_1.mp4
       ├── def_2.mp4
       └── int_1.mp4
```

## Acknowledgement
Our renderer is based on the [libgs](https://github.com/Awesome3DGS/LibGS) implementation of the [official](https://github.com/graphdeco-inria/gaussian-splatting) 3DGS repository.