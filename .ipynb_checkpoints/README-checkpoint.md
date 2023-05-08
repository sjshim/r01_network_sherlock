# network_fmri

**in progress** 


SETUP: 
Install Anaconda/Miniconda on Sherlock. 

Create environment `conda create -name tedana_env`
Install tedana `pip install tedana`
Install pydeface `pip install pydeface`
Install flywheel API `pip install flywheel-sdk`
flywheel cli (link to instructions)
ants path 


DOWNLOAD SCANS:
create file with scans to download using BIDS_scripts/create_downloadtxt.ipynb
download niftis to OAK using sbatches/fw_network.sbatch (~20minutes / particpant)

ORGANIZE:
BIDS raw niftis with discovery_BIDs.py (~20minutes / participant)
-potential place for error: conda errors with pydeface 

QA PROCESS: 
run sbatch/dev_fmriprep.sbatch (~2-3 days / participant)
run me_combination/me_combo.sbatch (~20minutes / participant)
BIDS me_combination/me_combo data using?
MRIQC sbatch/me_combo using dev_mriqc.batch
look at MRIQC files

PREPROCESS:
preprocess_

things that still need to be done: 
* BIDS diffusion and t2ws 
* move util functions to utils file to clean up scritps
* debug create_downloadtxt.ipynb with jaime
* beef up README & DOC strings
* parallelize tedana by task instead of by subject 
* setup dev_fmriprep by session 
* abstract paths, create paths.txt or something 
* create requirements.txt for environment setup (ask Jaime to be guinea pig) with version numbers
* cleanup flywheel download dir, write function to do after BIDS
* make metadata folder so that other folders aren't cluttered 
* set up .gitignore