# CUDA_VISIBLE_DEVICES=2,3 sh scripts/train.sh -g 2 -d ptv3_forest_segmentation -c ptv3_sementation_forest -n ptv3_sementation_forest 
sh scripts/train.sh -g 8 -d ptv3_forest_segmentation -c ptv3_decoder_sem_seg_dual_head -n ptv3_decoder_sem_seg_dual_head 
