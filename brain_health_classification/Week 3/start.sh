nohup python -u classify.py > output.txt & 
echo "PID: $!"
echo "Attaching the process via following command:"
echo "tail -f /proc/$!/fd/1"
tail -f /proc/$!/fd/1

