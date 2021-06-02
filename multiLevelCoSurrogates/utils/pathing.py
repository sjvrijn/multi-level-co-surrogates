from pathlib import Path

def guaranteeFolderExists(path_name):
    """ Make sure the given path exists after this call """
    path = Path(path_name).absolute()
    path.mkdir(parents=True, exist_ok=True)
