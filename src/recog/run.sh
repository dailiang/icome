#!/bin/sh
# auther @309 team
# Tel:   15019481435 
# Email: 986639399@qq.com
# Welcome to contact me if you encounter any problems
# date: 2013.10.01

if [ $# -ne 2 ]; then
    echo "Invalid number of arguments passed."
    echo "Correct Usage: ./run.sh /path/to/origin/pics/ /path/to/profile/pics/"
    echo ""
else
    (python process0.py $1 $2 &)
    (python process1.py $1 $2 &)
    (python process2.py $1 $2 &)
    python process3.py $1 $2 
fi
