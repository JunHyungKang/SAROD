base_dir_detections_fd = './data/256/base_dir_detections_fd' # Directory that contains detections by fine level detector
base_dir_detections_cd = './data/256/base_dir_detections_cd' # Directory that contains detections by coarse level detector
base_dir_groundtruth = './data/256/base_dir_groundtruth' # Directory that contains ground truth bounding boxes
base_dir_metric_fd = './data/256/base_dir_metric_fd' # Directory that contains AP or AR values by the fine detector
base_dir_metric_cd = './data/256/base_dir_metric_cd' # Directory that contains AP or AR values by the coarse detector
num_actions = 4 # Hyperparameter, should be equal to num_windows * num_windows
num_windows = 2 # Number of windows in one dimension
img_size_fd = 480 # Image size used to train the fine level detector
img_size_cd = 96 # Image size used to train the coarse level detector
