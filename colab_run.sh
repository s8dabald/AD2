#!/bin/bash

#start a synchronous colab run and stream output
#detached processes stop when the runtime ends
#copy test_results.xlsx to drive before the runtime clears
#push current code before cloning

S=ad2
#leave blank for cpu or set a gpu runtime, such as --gpu T4
#also enable task_type='GPU' in models/kittyboost.py
ACCEL=""

#stop the previous session if it exists
colab stop -s $S 2>/dev/null || true

colab new -s $S $ACCEL

#mount drive to preserve test_results.xlsx
colab drivemount -s $S

#clone the pushed code into a clean directory
echo '!cd /content && rm -rf AD2 && git clone https://github.com/s8dabald/AD2.git' | colab exec -s $S
colab install catboost

echo "Run started synchronously on $S. Streaming live output..."
echo '!cd /content/AD2 && python -u main.py' | colab exec -s $S

colab stop -s $S
