#!/bin/bash

######################################################

# This script performs the following:
# i) FastQC - extracts quality metrics from raw fastq files
# ii) Fastp - performs quality filtering and trimming on raw reads
# iii) PEAR - merges quality filtered / trimmed forward and reverse reads
# iv) Seqkit - extracts reads containing the desired barcode configuration
# outputs are written to ../../CombiCR_data/combiCR_outputs directory

### PARAMETERS
## $1: metadata in .txt format (tab separated)
# headers should include:
# sample_number (e.g. 1); experiment_name (e.g. maa-002);
# sample_name (e.g. 1_S1153_L001_R1_001.fastq.gz) - should match .fastq.gz file names;
# sample_type (e.g. single, double, triple), strain_name (e.g. YPH500),
# repressor name (e.g. tetR), reporter name (e.g. pMAA102),  inducer (e.g. aTc),
# bin number (e.g. 1); replicate number (e.g., 1 or 2); condition_id  (e.g. pMAA06-102_bin1_double_rep1)
## $2: list of fastq or fastq.gz files to be run through FastQC
# Only specify one of each read pair; the other
# filename is assumed (R2 if R1, or R1 if R2.)
# Example 1: sh 01_filter_merge_extract_reads.sh \
# ../../combicr_data/Quintata_NGS_raw/MAA_004/HiSeq_2022-10-26/Fastq/MAA_004_metadata.txt \
# ../../combicr_data/Quintata_NGS_raw/MAA_004/HiSeq_2022-10-26/Fastq/*_R1_001.fastq.gz
#
# Example 2: sh 01_filter_merge_extract_reads.sh \
# ../../combicr_data/Quintata_NGS_raw/MAA_005/AVITI_2023-10-23/Fastq/MAA_005_metadata.txt \
# ../../combicr_data/Quintata_NGS_raw/MAA_005/AVITI_2023-10-23/Fastq/*_R1_001.fastq.gz
#
### OUTPUTS
## FastQC
## $fastqc_out_path.html : fastqc report -- web link
## $fastqc_out_path.zip : fastqc report -- detailed files

## Fastp
## ${fastp_out_path}_out_R1.fastq.gz : quality filtered forward read
## ${fastp_out_path}_out_R2.fastq.gz  : quality filtered reverse read

## PEAR
## ${pear_out_path}_out_merged.assembled.fastq : merged fastq file that passed
# qc/qa filters

## Seqkit
## ${seqkit_stats_out_path}_filtering_stats.txt: statistics on number of reads
# containing barcode
## ${seqkit_filtered_reads_out_path}_out_merged_filtered.fastq: reads that have
# been merged (with PEAR) and contain a conserved sequence that indicates the read
# likely contains the barcode of interest
## ${seqkit_barcodes_out_path}: barcodes extracted for each read

## NOTES
# this script assumes the file name ends in R[12]_001.fastq.gz
# any string is allowed before R[12], as long as the forward and reverse read
# prefix is the same

######################################################

# activate virtual enviroment
source activate CombiCR

# load in metadata and remove from input
METADATA=$1
shift

# check to make sure files have the expected formats
for fastq in $@
do
	if [ -z $(basename $fastq | grep -i .fastq) ]
	then
		echo $(basename $fastq) "does not have .fastq suffix - aborting"
		exit 1
	fi
done

# loop through all fastq files and extract file names and metadata required for
# writing outputs to correct folder
for fastq in "$@"
do
	fname=$(basename $fastq)
	dname=$(dirname $fastq)
	fpath=$dname/${fname%_R[12]_001.fastq*}
	experiment_name=$(cat $METADATA | grep $fname | cut -f2)
  combination_type=$(cat $METADATA | grep $fname | cut -f4)

  echo Experiment name $experiment_name
	fastqc_out_path=../../combiCR_data/combiCR_outputs/$experiment_name/fastqc_output/
  fastp_out_path=../../combiCR_data/combiCR_outputs/$experiment_name/fastp_output/${fname%_R[12]_001.fastq*}
  pear_out_path=../../combiCR_data/combiCR_outputs/$experiment_name/merged_reads/${fname%_R[12]_001.fastq*}
	seqkit_stats_out_path=../../combiCR_data/combiCR_outputs/$experiment_name/seqkit_stats/${fname%_R[12]_001.fastq*}
	seqkit_filtered_reads_out_path=../../combiCR_data/combiCR_outputs/$experiment_name/merged_barcode_filtered_reads/${fname%_R[12]_001.fastq*}
	seqkit_barcodes_out_path=../../combiCR_data/combiCR_outputs/$experiment_name/barcodes/${fname%_R[12]_001.fastq*}
	echo Beginning $fname

	# define prefix for all output directories
	output_dir_prefix=../../combiCR_data/combiCR_outputs/$experiment_name

	if [ ! -e ${pear_out_path}_out_merged.assembled.fastq ]
	then

	  # create output directories if they don't exist already
		if [[ "$fastq" == "$1" ]]
			then

				# create general output folder for current experiment
				if [ ! -d "$output_dir_prefix" ]
			  then
					echo Making directory $output_dir_prefix
			    mkdir $output_dir_prefix
				fi

				# output directory for fastqc results
				if [ ! -d "$output_dir_prefix/fastqc_output/" ]
			  then
					echo Making directory $output_dir_prefix/fastqc_output/
			    mkdir $output_dir_prefix/fastqc_output/
				fi

				# output directory for fastp results
	      if [ ! -d "$output_dir_prefix/fastp_output/" ]
			  then
					echo Making directory $output_dir_prefix/fastp_output/
			    mkdir $output_dir_prefix/fastp_output/
				fi

				# output directory for merged reads
	      if [ ! -d "$output_dir_prefix/merged_reads/" ]
			  then
					echo Making directory $output_dir_prefix/merged_reads/
			    mkdir $output_dir_prefix/merged_reads/
				fi

				# output directory for seqkit stats
	      if [ ! -d "$output_dir_prefix/seqkit_stats/" ]
			  then
					echo Making directory $output_dir_prefix/seqkit_stats/
			    mkdir $output_dir_prefix/seqkit_stats/
				fi

				# output directory for merged reads that have been filtered for reads
				# containing the desired barcode configuration
				if [ ! -d "$output_dir_prefix/merged_barcode_filtered_reads/" ]
			  then
					echo Making directory $output_dir_prefix/merged_barcode_filtered_reads/
			    mkdir $output_dir_prefix/merged_barcode_filtered_reads/
				fi

				# output directory for barcodes in each
				if [ ! -d "$output_dir_prefix/barcodes/" ]
			  then
					echo $output_dir_prefix/barcodes/
			    mkdir $output_dir_prefix/barcodes/
				fi
		fi

	  if [[ "$combination_type" == "single" ]]
			then
				echo "Combination type is: single."
	      #expected amplicon is 203bp -- allowing for +/- 3bp error
	      min_merge_len=200
	      max_merge_len=206
				conserved_seq=gaaaaatTTTTTGGATCCGCAACGGAattc
				primer_F=TTGGATCCGCAACGGAattc
				primer_R=cagccaAACAACAACAATTg



		elif [[ "$combination_type" == "double" ]]
			then
				echo "Combination type is: double."
	      #expected amplicon is 193bp -- allowing for +/- 3bp error
	      min_merge_len=190
	      max_merge_len=196
				conserved_seq=TGTGCGTTTTTTGGATCCGCAACGGAattc
				primer_F=TTGGATCCGCAACGGAattc
				primer_R=cagccaAACAACAACAATTg

		else
			echo $strain "Combination type must be single or double -- aborting."
			exit 1
		fi

	  # run fastqc on forward and reverse reads
	  fastqc ${fpath}_R1_001.fastq.gz -o $fastqc_out_path
	  fastqc ${fpath}_R2_001.fastq.gz -o $fastqc_out_path

	  # run fastp (with default settings) on forward and reverse reads
		# save quality report
	  fastp \
	      -i ${fpath}_R1_001.fastq.gz \
	      -I ${fpath}_R2_001.fastq.gz \
	      -o ${fastp_out_path}_out_R1.fastq.gz \
	      -O ${fastp_out_path}_out_R2.fastq.gz \
	      -j ${fastp_out_path}.json \
	      -h ${fastp_out_path}.html

	  # run pear (i.e., merge) on quality filtered forward and reverse reads
	  pear \
	      -f ${fastp_out_path}_out_R1.fastq.gz \
	      -r ${fastp_out_path}_out_R2.fastq.gz \
	      -n $min_merge_len \
	      -m $max_merge_len \
	      -o ${pear_out_path}_out_merged \
				-j 6

		# remove additional pear output files that will not be used
	  rm ${pear_out_path}_out_merged.unassembled.forward.fastq
	  rm ${pear_out_path}_out_merged.unassembled.reverse.fastq
	  rm ${pear_out_path}_out_merged.discarded.fastq
	fi

	#### FILTER FASTQ FILES USING SEQKIT
	echo Beginning filtering for $fname

	# extract reads that contain the desired barcode configuration
	# (e.g., attcNNNNNNNNNcAattcNNNNNNNNNcAATT for doubles)
	cat ${pear_out_path}_out_merged.assembled.fastq | seqkit stats > ${seqkit_stats_out_path}_filtering_stats.txt
	cat ${pear_out_path}_out_merged.assembled.fastq | seqkit grep -s -i -m 2 -P -p $conserved_seq | seqkit stats >> ${seqkit_stats_out_path}_filtering_stats.txt
	cat ${pear_out_path}_out_merged.assembled.fastq | seqkit grep -s -i -m 2 -P -p $conserved_seq -o ${seqkit_filtered_reads_out_path}_out_merged_filtered.fastq
	cat ${seqkit_filtered_reads_out_path}_out_merged_filtered.fastq | seqkit amplicon -m 4 -P -s -F $primer_F -R $primer_R -r 21:44 --bed > ${seqkit_barcodes_out_path}_barcodes.txt
	# rm ${pear_out_path}_out_merged.assembled.fastq
  echo _____________new sample____________________________

done
