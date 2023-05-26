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

python make_depth_info.py
python make_image_info.py
python merge_event_data.py

if [ "$FLAG_M" = "TRUE" ];then
    echo "=========================================="
    echo " Generating meta data"
    echo "=========================================="

    mkdir -p ./list/train/
    mkdir -p ./list/val/
    mkdir -p ./list/test/

    ls ./train_evt/*.npy > ./list/train/events.txt
    ls ./train_img/*.npy > ./list/train/images.txt
    ls ./train_lbl/*.npy > ./list/train/labels.txt

    ls ./test_evt/*.npy > ./list/val/events.txt
    ls ./test_img/*.npy > ./list/val/images.txt
    ls ./test_lbl/*.npy > ./list/val/labels.txt

    ls ./test_evt/*.npy > ./list/test/events.txt
    ls ./test_img/*.npy > ./list/test/images.txt
    ls ./test_lbl/*.npy > ./list/test/labels.txt

    python ./scripts/save_video_duration.py
    python ./scripts/make_event_meta.py train
    python ./scripts/make_event_meta.py val
    python ./scripts/make_event_meta.py test
    python ./scripts/merge_meta.py
fi

