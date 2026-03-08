%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%		README -- CASE_dataset/data/initial
%
% This short guide to the initial data, covers the following topics:
% (1) Preamble and general information.
% (2) Usage.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-------------------------------------------------------------------------------
(1) Preamble and general information:
-------------------------------------------------------------------------------
The raw data acquired from each participant during the experiment is stored in
two different tab delimited text files. Where, one contains the physiological,
and the other, the annotation data. This was required because the the sampling
rates for the DAQ and annotation setups are different, i.e., 1000 Hz and 20 Hz,
respectively. Due to hardware restrictions, the sampling rate for annotation
joystick could not be set higher than 20 Hz.

Manipulating these large text files in MATLAB is however very slow. To this end,
the raw data in the annotation and physiological files for each subject (e.g.,
sub1_joystick.txt and sub1_DAQ.txt, respectively) is:
	(1) extracted from these text files, and then
	(2) saved in MATLAB preferred .mat format
using the script s02_extractSubData.m. This results in a single mat file 
containing both annotation and physiological data (e.g. sub_1.mat). This folder
contains the resulting mat files. These mat files have faster read and save
times when undertaking pre-processing in the subsequent steps.  

-------------------------------------------------------------------------------
(2) Usage:
-------------------------------------------------------------------------------
Since a lot of researchers use MATLAB, we decided to share these mat files
as a part of our dataset, so as to make it quick to get started.
 
Please note however, that the data here has not been pre-processed and therefore
does not contain any video-IDs. Nevertheless, the scripts provided with the
dataset implement all the necessary steps to convert the data in the initial
files to interpolated and non-interpolated data. 

Interested researchers can therefore modify the pre-processing scripts, namely 
s03_v1_tranformData.m and s03_v2_interTranformData.m, that load this
initial data such that the resulting processed data is also saved in mat files.
As previously mentioned, this would lead to have faster file-read and -save 
times when undertaking any downstream analyses.    
    

  
