# TARGET_HOSTNAME="jcollins90@hrl-ultron"
TARGET_HOSTNAME="patrick@143.215.225.230"

# rsync -aP ~/sd-video-gen/ $TARGET_HOSTNAME:~/sd-video-gen/
rsync -aP $TARGET_HOSTNAME:~/sd-video-gen/checkpoints ~/mnt/extdisk/sd-video-gen/checkpoints