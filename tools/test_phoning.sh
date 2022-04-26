#!/bin/bash
# Basic while loop
i=0
while [ $i -le 9 ]
do
	j=0
	while [ $j -le 9 ]
	do
		echo "{\"true threshold\": 0.${i}, \"false threshold\": 0.${j}}" > options.json
		python3 ~/brainframe/open_vision_capsules/tools/capsule_classifier_accuracy/classifier_accuracy.py --capsule /home/leefr/brainframe/pharmacy/private/classifier_phoning_factory_openvino --images-true /home/leefr/capsules-test/pictures/2022-04-25/test/phone --images-false /home/leefr/capsules-test/pictures/2022-04-25/test/nophone --data attribute=phoning true_threshold="0.$i" false_threshold="0.$j" detection=person
		#./capsule_classifier_accuracy_v0.3 --capsule /home/leefr/brainframe/pharmacy/private/classifier_phoning_factory_openvino --images-true /home/leefr/capsules-test/pictures/2022-04-25/test/phone --images-false /home/leefr/capsules-test/pictures/2022-04-25/test/nophone --data attribute=phoning true_threshold="0.$i" false_threshold="0.$j" detection=person
		((j++))
	done
	((i++))
done
echo All done
