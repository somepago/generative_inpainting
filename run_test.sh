#!/bin/bash

checkpoint_dir="./model_logs/robust"
image_dir="/cmlscratch/gowthami/data/celebahq/celebahq256"
# image_dir="/cmlscratch/gowthami/generative_inpainting/examples/celebahq"

# image_filelist="/cmlscratch/gowthami/data/celebahq/all_files.txt"
image_filelist="/cmlscratch/gowthami/data/celebahq/val_shuffled.txt"

mask="./examples/celebahq/temp/06984_mask.png"

output="/cmlscratch/gowthami/generative_inpainting/output/celebahq_robust"

epsilon=8


python test.py --checkpoint_dir=${checkpoint_dir} --image_dir=${image_dir} --image_filelist=${image_filelist} --mask=${mask} --output=${output} --epsilon=${epsilon}



# for epsilon in {2,4,8,16,32,64}
# do


# python test.py --checkpoint_dir=${checkpoint_dir} --image_dir=${image_dir} --image_filelist=${image_filelist} --mask=${mask} --output=${output} --epsilon=${epsilon}

# done
# echo All done