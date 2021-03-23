from src.app.io import get_train_root_dir
from src.core.common.utils import get_subdir


def get_train_dir(base_dir: str, train_name: str, create: bool):
  return get_subdir(get_train_root_dir(base_dir, "tacotron", create), train_name, create)

