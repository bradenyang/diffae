#!/usr/bin/env python

# ==========================
# ===== IMPORT MODULES =====
# ==========================

from templates import *

# ===================================
# ===== DEFINE GLOBAL VARIABLES =====
# ===================================

brats_dir = "/scratch/b.y.yang/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

train_df_path = "/scratch/b.y.yang/ESE5934-project/data/train.csv"
test_df_path = "/scratch/b.y.yang/ESE5934-project/data/test.csv"
model_ckpt_path = "/scratch/b.y.yang/ESE5934-project/diffae/checkpoints/last.ckpt"

device = "cuda"

# ============================
# ===== LOAD DATASET CSV =====
# ============================

dataset_df = pd.read_csv("/scratch/b.y.yang/ESE5934-project/data/dataset.csv")

# ==========================
# ===== CREATE DATASET =====
# ==========================

# define transforms without random flip/rotation
bbox_slice_row = slice(22, 216, None)
bbox_slice_col = slice(29, 223, None)
transform_seq = Compose([
    LoadImaged(keys = ["img"]),
    Squeeze2Dd(keys = ["img"]),
    SpatialCropd(keys = ["img"], roi_slices = [bbox_slice_row, bbox_slice_col, slice(None)]),
    ScaleIntensityd(keys = ["img"]),
    Resized(keys = ["img"], spatial_size = (128, 128), size_mode = "all"),  # resize to 1 x 128 x 128
    GrayscaleToRGBd(keys = ["img"]),
    ToTensord(keys = ["img"]),
])

# create dataset
cache_dir = mkdtemp(dir = "/scratch/b.y.yang")
data = PersistentDataset(
    data = [{"img": nii_path, "index": i} for i, nii_path in enumerate(dataset_df["slice_path"])],
    transform = transform_seq,
    cache_dir = cache_dir
)

# =================================
# ===== LOAD MODEL CHECKPOINT =====
# =================================

conf = brats_autoenc()
print(conf.name)
model = LitModel(conf)
state = torch.load(model_ckpt_path, map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);

# ====================================
# ===== ENCODE SEMANTIC FEATURES =====
# ====================================

latent_features = np.zeros((dataset_df.shape[0], 512))

for i, x in enumerate(data):
    
    # get image (note: None indexing adds an extra dimension to the front of the array dimensions)
    img = x["img"][None, :]

    # encode
    cond = model.encode(img.to(device))

    # store latent features as numpy array
    latent_features[i, :] = cond[0].cpu().numpy().squeeze()  

# store features in dataframe
col_names = np.char.add("latent", np.char.zfill((np.arange(512) + 1).astype(str), 3))
latent_df = pd.DataFrame(latent_features, columns = col_names)
dataset_df = pd.concat([dataset_df, latent_df], axis = 1)

# ==============================
# ===== SAVE NEW DATAFRAME =====
# ==============================

dataset_df.to_csv("/scratch/b.y.yang/ESE5934-project/data/dataset_latent.csv")
