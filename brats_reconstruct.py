from templates import *

import joblib

# load trained model
# device = 'cuda'
device = "cpu"
conf = brats_autoenc()
print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);

# define transforms
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

# define dataset
cache_dir = mkdtemp(dir = "/scratch/b.y.yang")
dataset_df = pd.read_csv("/scratch/b.y.yang/ESE5934-project/data/dataset.csv")
test_dataset_df = pd.read_csv("/scratch/b.y.yang/ESE5934-project/data/test.csv")
# dataset_df = dataset_df.iloc[:2, :]
# dataset = PersistentDataset(
#     data = [{"img": nii_path} for nii_path in dataset_df["slice_path"]],
#     transform = transform_seq,
#     cache_dir = cache_dir
# )
test_data = PersistentDataset(
    data = [{"img": nii_path} for nii_path in test_dataset_df["slice_path"]],
    transform = transform_seq,
    cache_dir = cache_dir
)

# select example images
# img_1 = train_data[100]["img"][None]
# img_2 = test_data[100]["img"][None]

recon_dict = {}
for i, img_idx in enumerate([165, 380, 550, 600]):
    # get image
    img = test_data[img_idx]["img"][None]

    # encode
    cond = model.encode(img.to(device))
    xT = model.encode_stochastic(img.to(device), cond, T=250)

    # decode
    pred = model.render(xT, cond, T=20)

    # store in dictionary
    img_dict = {}
    img_dict["orig"] = img
    img_dict["recon"] = pred
    recon_dict[img_idx] = img_dict

    # # plot
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # # ori = (batch + 1) / 2
    # ax[0].imshow(img[0].permute(1, 2, 0).cpu())
    # ax[1].imshow(pred[0].permute(1, 2, 0).cpu())
    # ax[0].set_xticks([]); ax[1].set_xticks([])
    # ax[0].set_yticks([]); ax[1].set_yticks([])
    # fig.savefig(f"/scratch/b.y.yang/ESE5934-project/figures/img{i}.png")

# joblib dump
joblib.dump(recon_dict, "/scratch/b.y.yang/ESE5934-project/data/recon_test.joblib")
