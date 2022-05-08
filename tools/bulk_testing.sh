#!/bin/bash

PROG_PREFIX=~/brainframe
DATA_PREFIX=~/capsules-test/pictures/2022-04-25/test
CAPSULE=$PROG_PREFIX/pharmacy/private/classifier_phoning_factory_openvino

IMAGE_TRUE_DIR=phone
IMAGE_FALSE_DIR=nophone
ATTRIBUTE=phoning
DETECTION=person

CAPSULE_INFER_PATH=$PROG_PREFIX/open_vision_capsules/tools/capsule_infer
CMD="python3 $PROG_PREFIX/open_vision_capsules/tools/capsule_classifier_accuracy/classifier_accuracy.py"
# CMD=capsule_classifier_accuracy_v0.3

ARGS="--capsule $CAPSULE --images-true $DATA_PREFIX/$IMAGE_TRUE_DIR --images-false $DATA_PREFIX/$IMAGE_FALSE_DIR --data attribute=$ATTRIBUTE detection=$DETECTION --nowait"

# Basic while loop
i=0
while [ $i -le 9 ]
do
	j=0
	while [ $j -le 9 ]
	do
		OPTIONS="{\"true threshold\": 0.${i}, \"false threshold\": 0.${j}}"
		echo $OPTIONS > options.json
		echo $OPTIONS
		echo $CMD $ARGS true_threshold="0.$i" false_threshold="0.$j"
		export PYTHONPATH=$PYTHONPATH:$CAPSULE_INFER_PATH
		$CMD $ARGS
		((j++))
	done
	((i++))
done
echo All done
