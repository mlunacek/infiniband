# Infiniband

## Meta

Create an `ibnetdiscover` file anytime the system is rebooted. From **moab**, type:

	ibnetdiscover /lustre/janus_scratch/molu8455/infiniband/meta/ibnetdiscover-<date>

Convert this file into a list linking *nodes* to *lids*.

	cd meta
	python python create_node_lid.py ibnetdiscover-2-12-2014

## Collect data

Each job will get a unique directory called test-$PBS_JOBID.  The results file will be named results.$PBS_JOBID.

Edit the `submit.pbs` script and type:
	
	qsub submit.pbs

Modify at submission time.

	qsub -I -l nodes=100:ppn=12 submit.pbs

## Parse results

Parse individual files using the `parse.py` script.

	python parse.py ../collect/test-<JOB_ID>/results.<JOB_ID>
	
Parse all the files in the `../collect/test-*` directory:

	python parse_all.py

The ouput directory is specified in the `config.py` file.

## Uniqueness

	python unique.py





