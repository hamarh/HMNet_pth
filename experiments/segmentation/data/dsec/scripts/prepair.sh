# Hierarchical Neural Memory Network
# 
# Copyright (C) 2023 National Institute of Advanced Industrial Science and Technology
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of {{ project }} nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

while getopts ah OPT
do
    case $OPT in
        "a" ) FLAG_M=TRUE;;
        "h" ) FLAG_H=TRUE;;
          * ) echo "Usage: $CMDNAME [-a] [-h]" 1>&2
              exit 1;;
    esac
done

if [ "$FLAG_H" = "TRUE" ];then
    echo "Usage: ${0##*/} [-m] [-h]"
    echo "    -m  : Generate meta data. This may take long time and requires large cpu memory (>32GB)"
    echo "    -h  : Show this message"
    echo ""
    exit 0
fi


echo "=========================================="
echo " Start preprocessing"
echo "=========================================="

TRAIN=(
    zurich_city_00_a
    zurich_city_01_a
    zurich_city_02_a
    zurich_city_04_a
    zurich_city_05_a
    zurich_city_06_a
    zurich_city_07_a
    zurich_city_08_a
)

TEST=(
    zurich_city_13_a
    zurich_city_14_c
    zurich_city_15_a
)

mkdir -p train_evt
mkdir -p train_img
mkdir -p train_lbl
mkdir -p train_img_right
mkdir -p test_evt
mkdir -p test_img
mkdir -p test_lbl
mkdir -p test_img_right
mkdir -p list


for dir in ${TRAIN[@]};do
    ln -s $(readlink -f ./source/$dir/events/left/events.h5) ./train_evt/${dir}_events.h5
    ln -s $(readlink -f ./source/$dir/images/left/rectified) ./train_img/${dir}_images
    ln -s $(readlink -f ./source/train/$dir/11classes) ./train_lbl/${dir}_labels
    ln -s $(readlink -f ./source/$dir/images/right/rectified) ./train_img_right/${dir}_images
done


for dir in ${TEST[@]};do
    ln -s $(readlink -f ./source/$dir/events/left/events.h5) ./test_evt/${dir}_events.h5
    ln -s $(readlink -f ./source/$dir/images/left/rectified) ./test_img/${dir}_images
    ln -s $(readlink -f ./source/test/$dir/11classes) ./test_lbl/${dir}_labels
    ln -s $(readlink -f ./source/$dir/images/right/rectified) ./test_img_right/${dir}_images
done

python ./scripts/make_image_info.py
python ./scripts/make_label_info.py
python ./scripts/make_image_info_right.py


python ./scripts/preproc_events.py ./train_evt/ --num_chunks 10
python ./scripts/preproc_events.py ./test_evt/ --num_chunks 10
python ./scripts/preproc_images_and_labels.py ./train_lbl/ --root ./ --input_type label --num_chunks 10
python ./scripts/preproc_images_and_labels.py ./train_img/ --root ./ --input_type image --num_chunks 10
python ./scripts/preproc_images_and_labels.py ./train_img_right/ --root ./ --input_type image --num_chunks 10
python ./scripts/preproc_images_and_labels.py ./test_lbl/ --root ./ --input_type label --num_chunks 10
python ./scripts/preproc_images_and_labels.py ./test_img/ --root ./ --input_type image --num_chunks 10
python ./scripts/preproc_images_and_labels.py ./test_img_right/ --root ./ --input_type image --num_chunks 10

if [ "$FLAG_M" = "TRUE" ];then
    echo "=========================================="
    echo " Generating meta data"
    echo "=========================================="

    mkdir -p ./list/train/
    mkdir -p ./list/test/
    ls ./train_evt/*.hdf5 > ./list/train/events.txt
    ls ./train_img/*.hdf5 > ./list/train/images.txt
    ls ./train_lbl/*.hdf5 > ./list/train/labels.txt
    ls ./train_img_right/*.hdf5 > ./list/train/images_right.txt

    ls ./test_evt/*.hdf5 > ./list/test/events.txt
    ls ./test_img/*.hdf5 > ./list/test/images.txt
    ls ./test_lbl/*.hdf5 > ./list/test/labels.txt
    ls ./test_img_right/*.hdf5 > ./list/test/images_right.txt

    python ./scripts/save_video_duration.py
    python ./scripts/make_event_meta.py train
    python ./scripts/make_event_meta.py test
    python ./scripts/merge_meta.py
fi




