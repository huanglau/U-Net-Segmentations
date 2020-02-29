# segmentation_class
Segmentation in keras in a medical imaging context using U-Net.

Includes unit tests for error metrics.

This pipeline will save the predictions in a designated experiment directory along with the code used and the versions of the libraries used. Error metrics (false negatives rate, false positive rate etc) will also be saved in the experiment directory along with error maps.

Currently using Sky dataset based on the Caltech Airplanes Side dataset (R. Fergus 15/02/03, http://www.robots.ox.ac.uk/~vgg/data3.html)
Datase can be found https://www.ime.usp.br/~eduardob/datasets/sky/.
