Data_Set Link :- https://zenodo.org/records/5511912

This data set contains measurements on birds, humans and six different drones using a frequency modulated continuous wave (FMCW) radar. The radar is a SAAB SIRS 1600 operating at 77 GHz with a mechanically scanning antenna.

Each sample is a scan segment of 5 samples in range (after range compression) and 256 samples in azimuth centered on the target. The segments are stored as a 1280 element complex-valued vector where the first 256 samples correspond to the first range cell and so forth.

The FMCW bandwidth is 160 MHz yielding a range resolution of about 1 m.
The radar PRF is 17kHz and scans 18 degrees in 70 ms, with 10Hz scan rate. The 256 azimuth samples thus correspond to 15 ms in time or ca. 3.9 degrees in azimuth.

The data is stored as a “.mat” file and contains a 130 by 6 cell matrix. The matrix can be loaded into python using:

import scipy.io
MatData=scipy.io.loadmat(<name of file>)
data=MatData['data']

Each of the 130 rows correspond to one measurement containing several scan segments, with a total of 75868 segments.   

The six columns contain the following data:

Column 1: string specifying data type and is one of the following

'D1' : Drone, DJI matrice 200 v
'D2' : Drone, DJI Mavic 2 Pro
'D3' : Drone, DJI Phantom 3
'D4' : Drone, Custom built FPV drone, see [1]
'D5' : Drone, Custom built with a Tarot 680 Pro frame, see [1]
'D6' : Drone, Syma X23W
'human_walk' : Different humans walking, ranging from almost still to “normal” walking.
'human_run' : Different humans running
'seagull' : Flying seagulls.
'pigeon' : Flying pigeon, very few samples
'raven' : Flying raven, very few samples
'black-headed gull' : Flying black-headed gulls
'seagull and black-headed gull' : Mixture of flying seagulls and black-headed gulls
'heron' : Flying heron
'CR' : Triangular corner reflector with 100m2 RCS at 77GHz
 

Column 2: Complex-valued data matrix of scan segments with 1280 rows and one column for each scan segment.

Column 3: Range in meters to center range cell (third range index).

Column 4: Time in seconds to segment center (azimuth index 128) with start at first FM-sweep in first scan in each measurement. Segments are stored in temporal order with an average of 100 ms between segments, however gaps exists when the target was either outside the field of view or when the return signal was indistinguable from noise . 

Column 5: Indicator of whether a sample is used in the training set, validation set, or test set in [1].
1=training
2=validation
3=test

Column 6: Binary indicator with logical one if the target is close to the field of view edge, zero otherwise.  Samples near edges are truncated in azimuth. In such cases, the “missing” azimuth sweeps are filled in  with noise. This is the case for 67 of the 75868 segment samples.

 

This data set is used in [1] where only FM-sweeps corresponding to azimuth index 54 to 203 are used, or 150 sweeps. 

When using this data set please refer to:

[1]  A. Karlsson, M. Jansson and M. Hämäläinen, "Model-Aided Drone Classification Using Convolutional Neural Networks," 2022 IEEE Radar Conference. contact: alek@kth.se