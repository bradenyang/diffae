# TODO
# - error in channels and tensor dimensions; use custom transform to reshape to correct length
# - reshape grayscale image to RGB; see:
#   - https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315
#   - https://discuss.pytorch.org/t/runtimeerror-given-groups-1-weight-64-3-3-3-so-expected-input-16-64-256-256-to-have-3-channels-but-got-64-channels-instead/12765
#   - https://discuss.pytorch.org/t/which-part-of-pytorch-tensor-represents-channels/21778

# ==========================
# ===== IMPORT MODULES =====
# ==========================

from shutil import rmtree

from templates import *
from templates_latent import *

# ====================================================================
# =============================== MAIN ===============================
# ====================================================================

if __name__ == '__main__':
    try:

        # train the autoenc moodel
        # this requires V100s.
        # gpus = [0, 1, 2, 3]
        gpus = [0]
        conf = brats_autoenc()
        conf.scale_up_gpus(1)

        train(conf, gpus=gpus)

        # # infer the latents for training the latent DPM
        # # NOTE: not gpu heavy, but more gpus can be of use!
        # # gpus = [0, 1, 2, 3]
        # gpus = [0, 1]
        # conf.eval_programs = ['infer']
        # train(conf, gpus=gpus, mode='eval')

        # # train the latent DPM
        # # NOTE: only need a single gpu
        # gpus = [0]
        # conf = ffhq128_autoenc_latent()
        # train(conf, gpus=gpus)

        # # unconditional sampling score
        # # NOTE: a lot of gpus can speed up this process
        # # gpus = [0, 1, 2, 3]
        # gpus = [0, 1]
        # conf.eval_programs = ['fid(10,10)']
        # train(conf, gpus=gpus, mode='eval')

    finally:
        # remove cache dir
        if os.path.isdir(conf.brats_cache_dir): rmtree(conf.brats_cache_dir, ignore_errors=True)