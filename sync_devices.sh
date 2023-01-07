# TARGET_HOSTNAME="jcollins90@hrl-ultron"
TARGET_HOSTNAME="patrick@143.215.225.230"

# rsync -aP ~/sd-video-gen/ $TARGET_HOSTNAME:~/sd-video-gen/
rsync -aP $TARGET_HOSTNAME:~/sd-video-gen/checkpoints/ ~/sd-video-gen/checkpoints/
# rsync -aP ~/sd-video-gen/data/UCF-101/ $TARGET_HOSTNAME:~/sd-video-gen/data/UCF-101/

# rsync -aP ~/sd-video-gen/checkpoints/11_27_ucf_text_final_0_test.pt $TARGET_HOSTNAME:~/sd-video-gen/checkpoints/
# rsync -aP ~/sd-video-gen/config/11_27_ucf_text_final.yml $TARGET_HOSTNAME:~/sd-video-gen/config/

# rsync -aP $TARGET_HOSTNAME:~/sd-video-gen/outputs_pred/ ~/sd-video-gen/outputs_pred/
# rsync -aP $TARGET_HOSTNAME:~/sd-video-gen/outputs_real/ ~/sd-video-gen/outputs_real/