ps -aux | grep xflow |grep -v auto| awk '{print $2}' | xargs  kill -9
