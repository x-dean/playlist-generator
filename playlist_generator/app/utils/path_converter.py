import os
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class PathConverter:
    """Convert paths between host and container filesystems."""

    def __init__(self, host_library: str = None, container_music: str = '/music'):
        self.host_library = host_library or os.getenv(
            'HOST_LIBRARY', '/root/music/library')
        self.container_music = container_music
        logger.debug(
            f"Initialized PathConverter: host_library={self.host_library}, container_music={self.container_music}")

    def host_to_container(self, host_path: str) -> str:
        """Convert a host path to container path."""
        logger.debug(f"Converting host path to container: {host_path}")

        try:
            if not host_path:
                logger.warning("Empty host path provided")
                return host_path

            # Normalize the host path
            host_path = os.path.normpath(host_path)
            logger.debug(f"Normalized host path: {host_path}")

            # Check if the path is already a container path
            if host_path.startswith(self.container_music):
                logger.debug(f"Path is already a container path: {host_path}")
                return host_path

            # Check if the path starts with the host library
            if not host_path.startswith(self.host_library):
                logger.warning(
                    f"Host path {host_path} doesn't start with {self.host_library}")
                return host_path

            # Convert host path to container path
            relative_path = os.path.relpath(host_path, self.host_library)
            container_path = os.path.join(self.container_music, relative_path)
            container_path = os.path.normpath(container_path)

            logger.debug(
                f"Converted host path {host_path} -> container path {container_path}")
            return container_path

        except Exception as e:
            logger.error(
                f"Error converting host path {host_path} to container: {str(e)}")
            import traceback
            logger.debug(
                f"Host to container conversion error traceback: {traceback.format_exc()}")
            return host_path

    def container_to_host(self, container_path: str) -> str:
        """Convert a container path to host path."""
        logger.debug(f"Converting container path to host: {container_path}")

        try:
            if not container_path:
                logger.warning("Empty container path provided")
                return container_path

            # Normalize the container path
            container_path = os.path.normpath(container_path)
            logger.debug(f"Normalized container path: {container_path}")

            # Check if the path is already a host path
            if container_path.startswith(self.host_library):
                logger.debug(f"Path is already a host path: {container_path}")
                return container_path

            # Check if the path starts with the container music directory
            if not container_path.startswith(self.container_music):
                logger.warning(
                    f"Container path {container_path} doesn't start with {self.container_music}")
                return container_path

            # Convert container path to host path
            relative_path = os.path.relpath(
                container_path, self.container_music)
            host_path = os.path.join(self.host_library, relative_path)
            host_path = os.path.normpath(host_path)

            logger.debug(
                f"Converted container path {container_path} -> host path {host_path}")
            return host_path

        except Exception as e:
            logger.error(
                f"Error converting container path {container_path} to host: {str(e)}")
            import traceback
            logger.debug(
                f"Container to host conversion error traceback: {traceback.format_exc()}")
            return container_path

    def is_container_path(self, path: str) -> bool:
        """Check if a path is a container path."""
        logger.debug(f"Checking if path is container path: {path}")

        try:
            if not path:
                logger.debug("Empty path provided")
                return False

            normalized_path = os.path.normpath(path)
            is_container = normalized_path.startswith(self.container_music)
            logger.debug(f"Path {path} is container path: {is_container}")
            return is_container

        except Exception as e:
            logger.error(f"Error checking if path is container path: {str(e)}")
            return False

    def is_host_path(self, path: str) -> bool:
        """Check if a path is a host path."""
        logger.debug(f"Checking if path is host path: {path}")

        try:
            if not path:
                logger.debug("Empty path provided")
                return False

            normalized_path = os.path.normpath(path)
            is_host = normalized_path.startswith(self.host_library)
            logger.debug(f"Path {path} is host path: {is_host}")
            return is_host

        except Exception as e:
            logger.error(f"Error checking if path is host path: {str(e)}")
            return False

    def get_relative_path(self, path: str) -> Optional[str]:
        """Get the relative path from either host or container path."""
        logger.debug(f"Getting relative path from: {path}")

        try:
            if not path:
                logger.warning("Empty path provided")
                return None

            normalized_path = os.path.normpath(path)

            # Try to get relative path from host library
            if normalized_path.startswith(self.host_library):
                relative_path = os.path.relpath(
                    normalized_path, self.host_library)
                logger.debug(
                    f"Got relative path from host library: {relative_path}")
                return relative_path

            # Try to get relative path from container music
            if normalized_path.startswith(self.container_music):
                relative_path = os.path.relpath(
                    normalized_path, self.container_music)
                logger.debug(
                    f"Got relative path from container music: {relative_path}")
                return relative_path

            logger.warning(
                f"Path {path} doesn't start with either host library or container music")
            return None

        except Exception as e:
            logger.error(f"Error getting relative path from {path}: {str(e)}")
            import traceback
            logger.debug(
                f"Get relative path error traceback: {traceback.format_exc()}")
            return None

    def normalize_path(self, path: str, target_type: str = 'auto') -> str:
        """Normalize a path to either host or container format.

        Args:
            path: The path to normalize
            target_type: 'host', 'container', or 'auto' (default)
        """
        logger.debug(f"Normalizing path: {path} (target_type={target_type})")

        try:
            if not path:
                logger.warning("Empty path provided")
                return path

            if target_type == 'host':
                if self.is_container_path(path):
                    return self.container_to_host(path)
                return path
            elif target_type == 'container':
                if self.is_host_path(path):
                    return self.host_to_container(path)
                return path
            else:  # auto
                # Determine the target type based on the current environment
                if os.path.exists(self.host_library):
                    # We're in a host environment, convert to host
                    if self.is_container_path(path):
                        return self.container_to_host(path)
                    return path
                else:
                    # We're in a container environment, convert to container
                    if self.is_host_path(path):
                        return self.host_to_container(path)
                    return path

        except Exception as e:
            logger.error(f"Error normalizing path {path}: {str(e)}")
            import traceback
            logger.debug(
                f"Normalize path error traceback: {traceback.format_exc()}")
            return path

    def validate_path(self, path: str) -> bool:
        """Validate that a path exists and is accessible."""
        logger.debug(f"Validating path: {path}")

        try:
            if not path:
                logger.warning("Empty path provided for validation")
                return False

            normalized_path = os.path.normpath(path)
            exists = os.path.exists(normalized_path)
            logger.debug(f"Path {path} exists: {exists}")
            return exists

        except Exception as e:
            logger.error(f"Error validating path {path}: {str(e)}")
            return False

    def get_path_info(self, path: str) -> dict:
        """Get information about a path."""
        logger.debug(f"Getting path info for: {path}")

        try:
            if not path:
                logger.warning("Empty path provided for path info")
                return {}

            normalized_path = os.path.normpath(path)
            info = {
                'original_path': path,
                'normalized_path': normalized_path,
                'exists': os.path.exists(normalized_path),
                'is_file': os.path.isfile(normalized_path) if os.path.exists(normalized_path) else False,
                'is_dir': os.path.isdir(normalized_path) if os.path.exists(normalized_path) else False,
                'is_container_path': self.is_container_path(normalized_path),
                'is_host_path': self.is_host_path(normalized_path),
                'relative_path': self.get_relative_path(normalized_path)
            }

            if os.path.exists(normalized_path):
                try:
                    info['size'] = os.path.getsize(normalized_path)
                except OSError:
                    info['size'] = None

                try:
                    info['modified'] = os.path.getmtime(normalized_path)
                except OSError:
                    info['modified'] = None

            logger.debug(f"Path info for {path}: {info}")
            return info

        except Exception as e:
            logger.error(f"Error getting path info for {path}: {str(e)}")
            import traceback
            logger.debug(
                f"Get path info error traceback: {traceback.format_exc()}")
            return {}

    def convert_playlist_tracks(self, container_tracks: List[str]) -> List[str]:
        """Convert container paths in playlist tracks to host paths."""
        logger.debug(f"Converting {len(container_tracks)} playlist tracks from container to host")
        
        host_tracks = []
        for track in container_tracks:
            try:
                if track.startswith('/music'):
                    # Convert container path to host path
                    host_path = self.container_to_host(track)
                    host_tracks.append(host_path)
                    logger.debug(f"Converted {track} -> {host_path}")
                else:
                    # Already a host path or relative path
                    host_tracks.append(track)
                    logger.debug(f"Track already in host format: {track}")
            except Exception as e:
                logger.warning(f"Error converting track {track}: {e}")
                # Keep original path if conversion fails
                host_tracks.append(track)
        
        logger.debug(f"Converted {len(host_tracks)} tracks to host paths")
        return host_tracks

    def convert_failed_files(self, container_failed_files: List[str]) -> List[str]:
        """Convert container paths in failed files list to host paths."""
        logger.debug(f"Converting {len(container_failed_files)} failed files from container to host")
        
        host_failed_files = []
        for failed_file in container_failed_files:
            try:
                if failed_file.startswith('/music'):
                    # Convert container path to host path
                    host_path = self.container_to_host(failed_file)
                    host_failed_files.append(host_path)
                    logger.debug(f"Converted failed file {failed_file} -> {host_path}")
                else:
                    # Already a host path or relative path
                    host_failed_files.append(failed_file)
                    logger.debug(f"Failed file already in host format: {failed_file}")
            except Exception as e:
                logger.warning(f"Error converting failed file {failed_file}: {e}")
                # Keep original path if conversion fails
                host_failed_files.append(failed_file)
        
        logger.debug(f"Converted {len(host_failed_files)} failed files to host paths")
        return host_failed_files
