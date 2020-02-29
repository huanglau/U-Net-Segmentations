#!/bin/bash
for i in {0..1}
do
	python foldsetup_configReader.py "config_segmentation.ini" $i 2
done
