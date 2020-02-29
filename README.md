# segmentation_class
Segmentation in keras in a medical imaging context using U-Net.

Includes unit tests for error metrics.

This pipeline will save the predictions in a designated experiment directory along with the code used and the versions of the libraries used. Error metrics (false negatives rate, false positive rate etc) will also be saved in the experiment directory along with error maps.

Currently using Sky dataset based on the Caltech Airplanes Side dataset (R. Fergus 15/02/03, http://www.robots.ox.ac.uk/~vgg/data3.html)
Datase can be found https://www.ime.usp.br/~eduardob/datasets/sky/.

To run this code through the command line or terminal navigate to the directory with the code.
You should be in the same directory where 'foldsetup_configReader.py' is stored.

Then you can run the bash script './NFold.sh' in unix systems or 'NFold.bat' in windows
systems.

This will run the experiment specified in 'config_segmentation.ini'. This
file contains the number of epochs, the type of model, augmentations, the directories to 
the training images etc. This config file is setup to output the exeperiment '../../Output/PlaneSegmentation'
so up two directories from where /src/ is.

If you want to use different data, you can create a new 'config_segmentation.ini'. You would update 
'Input File = SegmentationDataset.txt' in the file to either point to a different txt file, or 
change 'SegmentationDataset.txt' itself. 'SegmentationDataset.txt' is a txt file that lists the 
directories each 'group' is stored in. This is based on the idea that in medical 
applications you would often want to keep information of the same patient together.
By splitting the data into the groups, we insure that patients the same
patient will not appear in the training and testing dataset.