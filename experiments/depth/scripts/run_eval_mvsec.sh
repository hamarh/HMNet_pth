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

if [ $# -le 0 ];then
    echo "Usage: $0 [1]"
    echo "    [1]: config file"
    exit
fi

NAME=${1##*/}
NAME=${NAME%.py}

MAX_DEPTH=80
MIN_DEPTH=1.978
CUTOFF="30 20 10"

for data in day1 night1;do
    dir=$(ls -d ./workspace/${NAME}/result/pred_${data}/)
    pred=${dir}/outdoor_${data}_data.npy
    gt=./data/mvsec/source/outdoor_${data}_gt.hdf5
    out=${dir}/logs
    python ./scripts/eval_depth.py ${pred} ${gt} ${out} --max_depth ${MAX_DEPTH} --min_depth ${MIN_DEPTH} --cutoff_depth ${CUTOFF} --input_type mvsec
done

