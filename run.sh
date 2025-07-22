# Validate playlist method
case "$PLAYLIST_METHOD" in
    all|time|kmeans|cache|tags)
        ;;
    *)
        echo "Invalid playlist method: $PLAYLIST_METHOD"
        echo "Valid options are: all, time, kmeans, cache, tags"
        exit 1
        ;;
esac 