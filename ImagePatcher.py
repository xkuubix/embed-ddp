import torch
import numpy as np
from sklearn.utils import shuffle
import logging
logger = logging.getLogger(__name__)

class ImagePatcher:
    def __init__(self, patch_size=224, overlap=0.5, bag_size=-1, empty_thresh=0.8):
        self.patch_size = patch_size
        self.overlap = overlap
        self.bag_size = bag_size
        self.empty_thresh = empty_thresh
        self.tiles = None
        logger.info(f"ImagePatcher initialized with patch_size={patch_size}, overlap={overlap}, bag_size={bag_size}, empty_thresh={empty_thresh}")

    def _start_points(self, size, split_size):
        points = [0]
        stride = int(split_size * (1 - self.overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points

    def get_tiles(self, h, w):
        X_points = self._start_points(w, self.patch_size)
        Y_points = self._start_points(h, self.patch_size)
        
        tiles = np.zeros((len(Y_points) * len(X_points), 6), int)
        k = 0
        for i, y_cord in enumerate(Y_points):
            for j, x_cord in enumerate(X_points):
                tiles[k] = (y_cord, x_cord, self.patch_size, self.patch_size, i, j)
                k += 1
        self.tiles = tiles
        return tiles

    def convert_img_to_bag(self, image):
        if isinstance(image, np.ndarray):
            image =  torch.tensor(image, dtype=torch.float32)
        tiles = self.tiles
        hTile, wTile = tiles[0][2], tiles[0][3]
        c = image.shape[0]
        img_shape = (len(tiles), c, hTile, wTile)
        new_img = torch.zeros(img_shape)
        px_non_zero = torch.zeros(len(tiles), dtype=torch.float32)
        
        for i, tile in enumerate(tiles):
            new_img[i] = image[:, tile[0]:tile[0]+tile[2], tile[1]:tile[1]+tile[3]]
            px_non_zero[i] = (new_img[i][0] > 0).float().mean() * 100
        
        sorted_tiles_idx = np.argsort(-px_non_zero)
        px_non_zero_pc = (px_non_zero > self.empty_thresh * 100).sum()
        
        instances, instances_idx, instances_cords = self._select_bag(new_img, tiles, sorted_tiles_idx, px_non_zero_pc)
        return instances, instances_idx, instances_cords

    def reconstruct_image_from_patches(self, patches, instances_ids, image_shape):
        tiles = self.tiles
        c, h, w = image_shape
        
        reconstructed_image = torch.zeros(c, h, w, dtype=torch.float, device='cpu')
        patch_count = torch.zeros(c, h, w, dtype=torch.float, device='cpu')

        for item in range(len(instances_ids)):
            h_min, w_min, dh, dw, _, _ = tiles[instances_ids[item]]
            patch = patches[item]

            reconstructed_image[:, h_min:h_min + dh, w_min:w_min + dw] += patch
            patch_count[:, h_min:h_min + dh, w_min:w_min + dw] += 1

        patch_count = torch.where(patch_count == 0, torch.ones_like(patch_count), patch_count)
        return reconstructed_image

    def reconstruct_attention_map(self, attention_weights, instances_ids, image_shape):
        tiles = self.tiles
        n_passes, _, n_classes, n_patches = attention_weights.shape  # (n_passes, batch_size, n_classes, n_patches)
        c = 1
        h, w = image_shape

        reconstructed_attention = torch.zeros((n_passes, n_classes, c, h, w), dtype=attention_weights.dtype, device=attention_weights.device)
        patch_count = torch.zeros((n_passes, n_classes, c, h, w), dtype=torch.uint8, device=attention_weights.device)

        for item in range(n_patches):
            h_min, w_min, dh, dw, _, _ = tiles[instances_ids[item]]

            patch_attention = attention_weights[:, :, :, item]  # shape: (n_passes, batch_size, n_classes)
            patch_attention = patch_attention.view(n_passes, n_classes, 1, 1, 1)  # Expand to match spatial dims
            patch_attention = patch_attention.expand(n_passes, n_classes, c, dh, dw)

            reconstructed_attention[:, :, :, h_min:h_min + dh, w_min:w_min + dw] += patch_attention
            patch_count[:, :, :, h_min:h_min + dh, w_min:w_min + dw] += 1

        # avoiding division by zero
        patch_count = torch.where(patch_count == 0, torch.ones_like(patch_count), patch_count)

        # normalizing attention maps??
        reconstructed_attention /= patch_count
        if reconstructed_attention.min() < 0:
            reconstructed_attention = reconstructed_attention.abs()
        max_values = reconstructed_attention.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
        reconstructed_attention = reconstructed_attention / max_values.view(n_passes, n_classes, 1, 1, 1)

        return reconstructed_attention  # shape: (n_passes, n_classes, c, h, w)

    def _select_bag(self, new_img, tiles, sorted_tiles_idx, px_non_zero_pc):
        if self.bag_size > 0:
            if self.bag_size > px_non_zero_pc:
                instances = new_img[sorted_tiles_idx[:px_non_zero_pc]]
                instances_idx = sorted_tiles_idx[:px_non_zero_pc]
            else:
                instances = new_img[sorted_tiles_idx[:self.bag_size]]
                instances_idx = sorted_tiles_idx[:self.bag_size]
        elif self.bag_size == -1:
            instances = new_img[sorted_tiles_idx[:px_non_zero_pc]]
            instances_idx = sorted_tiles_idx[:px_non_zero_pc]
        else:
            raise ValueError("Invalid bag size")
        
        instances_cords = tiles[instances_idx, 4:6]
        return shuffle(instances, instances_idx, instances_cords)
