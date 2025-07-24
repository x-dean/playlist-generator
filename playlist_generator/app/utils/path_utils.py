import os

def convert_to_host_path(container_path: str, host_music_dir: str, container_music_dir: str) -> str:
    """Converts a path from the container to the host.

    Args:
        container_path (str): Path within the container.
        host_music_dir (str): Path to the host's music directory.
        container_music_dir (str): Path to the container's music directory.

    Returns:
        str: Path on the host.
    """
    container_path = os.path.normpath(container_path)
    container_music_dir = os.path.normpath(container_music_dir)
    if not container_path.startswith(container_music_dir):
        return container_path
    rel_path = os.path.relpath(container_path, container_music_dir)
    return os.path.join(host_music_dir, rel_path) 