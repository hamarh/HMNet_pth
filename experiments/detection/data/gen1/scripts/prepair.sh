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
    echo "    -m  : Generate meta data. This may take long time."
    echo "    -h  : Show this message"
    echo ""
    exit 0
fi


echo "=========================================="
echo " Start preprocessing"
echo "=========================================="

mkdir -p val_evt
mkdir -p test_evt
mkdir -p train_evt
mkdir -p val_lbl
mkdir -p test_lbl
mkdir -p train_lbl

python ./scripts/modify_lbl_field_name.py ./source/detection_dataset_duration_60s_ratio_1.0/val/ ./val_lbl/
python ./scripts/modify_lbl_field_name.py ./source/detection_dataset_duration_60s_ratio_1.0/test/ ./test_lbl/
python ./scripts/modify_lbl_field_name.py ./source/detection_dataset_duration_60s_ratio_1.0/train/ ./train_lbl/

python ./scripts/preproc_events.py train
python ./scripts/preproc_events.py val
python ./scripts/preproc_events.py test

python ./scripts/validate_bbox.py ./val_lbl/ ./val_lbl/
python ./scripts/validate_bbox.py ./test_lbl/ ./test_lbl/
python ./scripts/validate_bbox.py ./train_lbl/ ./train_lbl/

if [ "$FLAG_M" = "TRUE" ];then
    echo "=========================================="
    echo " Generating meta data"
    echo "=========================================="

    mkdir -p ./list/train/
    mkdir -p ./list/val/
    mkdir -p ./list/test/
    ls ./train_evt/*.npy > ./list/train/events.txt
    ls ./train_lbl/*.npy > ./list/train/labels.txt
    ls ./val_evt/*.npy > ./list/val/events.txt
    ls ./val_lbl/*.npy > ./list/val/labels.txt
    ls ./test_evt/*.npy > ./list/test/events.txt
    ls ./test_lbl/*.npy > ./list/test/labels.txt

    python ./scripts/make_event_meta.py train
    python ./scripts/make_event_meta.py val
    python ./scripts/make_event_meta.py test
    python ./scripts/merge_meta.py
    python ./scripts/get_gt_interval.py
fi


