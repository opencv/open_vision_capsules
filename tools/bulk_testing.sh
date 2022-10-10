#!/bin/bash

PROG_PREFIX=~/brainframe
DATA_PREFIX=~/dataset/dataset-2022-06-03
CAPSULE=$PROG_PREFIX/pharmacy/private/classifier_phoning_factory_openvino

IMAGE_TRUE_DIR=val_phone
IMAGE_FALSE_DIR=val_nophone
ATTRIBUTE=phoning
DETECTION=person

CAPSULE_INFER_PATH=$PROG_PREFIX/open_vision_capsules/tools/capsule_infer
CMD="python3 $PROG_PREFIX/open_vision_capsules/tools/capsule_classifier_accuracy/capsule_classifier_accuracy.py"
# CMD=capsule_classifier_accuracy_v0.3

BASIC_ARGS="--capsule $CAPSULE --images-true $DATA_PREFIX/$IMAGE_TRUE_DIR --images-false $DATA_PREFIX/$IMAGE_FALSE_DIR --nowait --data attribute=$ATTRIBUTE detection=$DETECTION "

# Basic while loop
i=3
while [ $i -le 9 ]
do
	j=0
	while [ $j -le 9 ]
	do
		OPTIONS="{\"true threshold\": 0.${i}, \"false threshold\": 0.${j}}"
		ARGS="$BASIC_ARGS true_threshold=0.$i false_threshold=0.$j"
		echo $OPTIONS > options.json
		echo $OPTIONS
		echo $CMD $ARGS
		export PYTHONPATH=$PYTHONPATH:$CAPSULE_INFER_PATH
		$CMD $ARGS
		((j++))
	done
	((i++))
done
echo All done
