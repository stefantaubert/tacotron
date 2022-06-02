import logging
from typing import List, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def calculate_structual_similarity_np(img_a: np.ndarray, img_b: np.ndarray) -> Tuple[float, np.ndarray]:
  #img_b = imageio.imread(path_original_plot)
  have_same_height = img_a.shape[0] == img_b.shape[0]
  have_same_width = img_a.shape[1] == img_b.shape[1]
  assert have_same_height and have_same_width
  score, diff_img = structural_similarity(
      im1=img_a,
      im2=img_b,
      full=True,
      channel_axis=-1,
  )
  #imageio.imsave(path_out, diff)
  # to prevent -> "WARNING:imageio:Lossy conversion from float64 to uint8. Range [-0.9469735935228797, 1.0000000000019036]."
  #diff_img = diff_img.astype(np.uint8)
  return score, diff_img


def stack_images_vertically(list_im, out_path) -> None:
  old_level = logging.getLogger().level
  logging.getLogger().setLevel(logging.INFO)
  images = [Image.open(i) for i in list_im]
  widths, heights = zip(*(i.size for i in images))

  total_height = sum(heights)
  max_width = max(widths)

  new_im = Image.new(
      mode='RGB',
      size=(max_width, total_height),
      color=(255, 255, 255)  # white
  )

  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]
  new_im.save(out_path)
  logging.getLogger().setLevel(old_level)


def stack_images_horizontally(list_im: List[str], out_path: str) -> None:
  old_level = logging.getLogger().level
  logging.getLogger().setLevel(logging.INFO)
  images = [Image.open(i) for i in list_im]
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new(
      mode='RGB',
      size=(total_width, max_height),
      color=(255, 255, 255)  # white
  )

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0]
  new_im.save(out_path)
  logging.getLogger().setLevel(old_level)
