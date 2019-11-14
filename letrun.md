bg %1

SAVED_PID=$(jobs -l | grep -v grep | awk '{print $2}')

echo $SAVED_PID was the pid

disown %1

alias letrun='bg %1 && SAVED_PID=$(jobs -l | grep -v grep | awk '{print $2}') && disown %1 && screen -dm bash -c 'kill -CONT "$SAVED_PID"''