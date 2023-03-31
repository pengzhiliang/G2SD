
import logging
import os
import pickle
import torch
from fvcore.common.checkpoint import Checkpointer
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from scipy import interpolate
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager

from .c2_model_loading import align_and_update_state_dicts


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to:
    1. handle models in detectron & detectron2 model zoo, and apply conversions for legacy models.
    2. correctly load checkpoints that are only available on the master worker
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.path_manager = PathManager

    def load(self, path, *args, **kwargs):
        need_sync = False

        if path and isinstance(self.model, DistributedDataParallel):
            logger = logging.getLogger(__name__)
            path = self.path_manager.get_local_path(path)
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                logger.warning(
                    f"Not all workers can read checkpoint {path}. "
                    "Training may fail to fully resume."
                )
                # TODO: broadcast the checkpoint file contents from main
                # worker, and load from it instead.
                need_sync = True
            if not has_file:
                path = None  # don't load if not readable
        ret = super().load(path, *args, **kwargs)

        if need_sync:
            logger.info("Broadcasting model states from main worker ...")
            self.model._sync_params_and_buffers()
        return ret

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
        elif filename.endswith(".pyth"):
            # assume file is from pycls; no one else seems to use the ".pyth" extension
            with PathManager.open(filename, "rb") as f:
                data = torch.load(f)
            assert (
                "model_state" in data
            ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
            model_state = {
                k: v
                for k, v in data["model_state"].items()
                if not k.endswith("num_batches_tracked")
            }
            return {"model": model_state, "__author__": "pycls", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        loaded["matching_heuristics"] = True
        return loaded

    def _load_model(self, checkpoint):
        # import pdb; pdb.set_trace()
        if 'module' in checkpoint and 'model' not in checkpoint:
            # checkpoint["model"] = checkpoint["module"].copy()
            new_dict = OrderedDict()
            for k, v in checkpoint["module"].items():
                new_dict[k] = v
            checkpoint["model"] = new_dict
        
        if 'model' in checkpoint:
            all_keys = list(checkpoint["model"].keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('module.'):
                    new_dict[key[7:]] = checkpoint["model"][key]
                else:
                    new_dict[key] = checkpoint["model"][key]
            checkpoint["model"] = new_dict
            
        if "rel_pos_bias.relative_position_bias_table" in checkpoint["model"].keys():
            checkpoint_state = self.process_relative_pos_emd_for_beit(self.model.state_dict(), checkpoint["model"])
            checkpoint["model"] = checkpoint_state
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            checkpoint["model"] = align_and_update_state_dicts(
                self.model.state_dict(),
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        for k in incompatible.unexpected_keys[:]:
            # Ignore unexpected keys about cell anchors. They exist in old checkpoints
            # but now they are non-persistent buffers and will not be in new checkpoints.
            if "anchor_generator.cell_anchors" in k:
                incompatible.unexpected_keys.remove(k)
        return incompatible

    def process_relative_pos_emd_for_beit(self, model_state, checkpoint_state):
        logger = logging.getLogger(__name__)
        checkpoint_rel_pos_bias = checkpoint_state.pop("rel_pos_bias.relative_position_bias_table")
        checkpoint_rel_pos_bias = checkpoint_rel_pos_bias[:-3, :] # exclude extra tokens

        src_num_pos, num_attn_heads = checkpoint_rel_pos_bias.size()
        _ = checkpoint_state.pop("rel_pos_bias.relative_position_index")

        model_all_keys = list(model_state.keys())
        for k in model_all_keys:
            # backbone.net.blocks.0.attn.relative_position_bias_table
            if "relative_position_bias_table" in k:
                dst_num_pos, _ = model_state[k].size()
                if src_num_pos == dst_num_pos:
                    checkpoint_state[k[len("backbone.net."):]] = checkpoint_rel_pos_bias.clone()
                else:
                    src_size = int(src_num_pos ** 0.5)
                    dst_size = int(dst_num_pos ** 0.5)
                    logger.info("Position interpolate for %s from %dx%d to %dx%d" % (
                        k, src_size, src_size, dst_size, dst_size))

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = checkpoint_rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(checkpoint_rel_pos_bias.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_state[k[len("backbone.net."):]] = new_rel_pos_bias

        return checkpoint_state