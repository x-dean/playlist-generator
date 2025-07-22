#!/bin/bash
set -e

# Get UID/GID from environment or use defaults
USER_ID=${CURRENT_UID:-1000}
GROUP_ID=${CURRENT_GID:-1000}

# Create group if it doesn't exist
if ! getent group $GROUP_ID > /dev/null; then
    groupadd -g $GROUP_ID appgroup
fi

# Create user if it doesn't exist
if ! getent passwd $USER_ID > /dev/null; then
    useradd -u $USER_ID -g $GROUP_ID -d /app -s /bin/bash appuser
fi

# Ensure directory permissions
mkdir -p /app/cache/checkpoints /app/cache/metrics /app/playlists
chown -R $USER_ID:$GROUP_ID /app/cache /app/playlists
chmod -R 755 /app/cache /app/playlists

# Silence Essentia info logs
export ESSENTIA_LOGGING_LEVEL=warning
export ESSENTIA_STREAM_LOGGING=none

# Run command as appuser
exec gosu $USER_ID:$GROUP_ID "$@" 