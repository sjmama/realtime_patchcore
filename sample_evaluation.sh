datapath=/2T/patchcore-inspection/mvtec
loadpath=/2T/patchcore-inspection/models
loadpath=/2T/patchcore-inspection/results/modified
modelfolder=mal
# modelfolder=IM320_Ensemble_L2-3_P001_D1024-1024_PS-5_AN-1
savefolder=evaluated_results'/'$modelfolder

datasets1=('grid')
datasets2=('grid')
model_flags=($(for dataset in "${datasets1[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets2[@]}"; do echo '-d '$dataset; done))
echo ${model_flags[3]}

#python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 --save_segmentation_images $savefolder \
#python bin/joljak.py --gpu 0 --seed 0 --save_segmentation_images $savefolder \
python bin/Joljak.py --gpu 0 --seed 0 --save_segmentation_images $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath