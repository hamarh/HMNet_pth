# Hierarchical Neural Memory Network
# 
# Copyright (C) 2023 National Institute of Advanced Industrial Science and Technology
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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




