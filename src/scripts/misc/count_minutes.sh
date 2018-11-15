#!/bin/bash

for fname in data/raw/StephenColbert/all/*.wav; do
	ffmpeg -i $fname 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | sed 's@\..*@@g' | awk '{ split($1, A, ":"); split(A[3], B, "."); print 3600*A[1] + 60*A[2] + B[1] }'
done

