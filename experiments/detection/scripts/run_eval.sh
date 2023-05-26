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

dir=$(ls -d ./workspace/${NAME}/result/pred_test/)
out=${dir}/logs
mkdir -p ${out}
python ./scripts/psee_evaluator.py ./data/gen1/test_lbl/ ${dir} --camera GEN1 > ${out}/result.txt
