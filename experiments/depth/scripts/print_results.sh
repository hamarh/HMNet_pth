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

args=("$@")
list=$(ls ./workspace/*/result/pred_day1/logs/result.txt)
if [ ${#list[@]} -ge 1 ];then
    echo -e "\n[outdoor day1]"
    python ./scripts/print_depth.py ${list[@]} ${args[@]}
fi
list=$(ls ./workspace/*/result/pred_night1/logs/result.txt)
if [ ${#list[@]} -ge 1 ];then
    echo -e "\n[outdoor night1]"
    python ./scripts/print_depth.py ${list[@]} ${args[@]}
fi
