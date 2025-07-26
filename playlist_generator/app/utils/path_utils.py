import os


def convert_to_host_path(container_path: str, library: str, music: str) -> str:
    """Converts a path from the container to the library directory.

    Args:
        container_path (str): Path within the container.
        library (str): Path to the user's music library directory.
        music (str): Path to the container's music directory.

    Returns:
        str: Path on the library.
    """
    container_path = os.path.normpath(container_path)
    music = os.path.normpath(music)
    if not container_path.startswith(music):
        return container_path
    rel_path = os.path.relpath(container_path, music)
    return os.path.join(library, rel_path)
