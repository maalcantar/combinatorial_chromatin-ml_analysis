# A high-throughput approach for studying combinatorial chromatin-based transcriptional regulation
## Introduction 

This repository contains all code needed to reproduce DNA sequencing data processing and analyses described in:
> Miguel A. Alcantar, Max A. English*, Jacqueline A. Valeri*, and James J. Collins. A high-throughput approach for studying combinatorial chromatin-based transcriptional regulation. <i>In revision</i>.

Code author: Miguel A. Alcantar.
# Installation & requirements 

This repository, including all code needed to reproduce analyses, can be installed using:

~~~
git clone https://github.com/maalcantar/combicr_analysis
cd combicr_analysis
pip install -r requirements_CombiCR.txt #requirements python for scripts 0-6; do this in a virtual environment called "CombiCR"
pip install -r requirements_combicr_ml.txt #requirements python for scripts 7-10; do this in a virtual environment called "combicr_ml"
~~~

R library requirements:
* DNABarcodes v1.32.0 #(https://bioconductor.org/packages/release/bioc/html/DNABarcodes.html)
* NuPoP v2.10.0 #(https://bioconductor.org/packages/release/bioc/html/NuPoP.html)

Additional requirements: 
* fastp v0.12.4 #(https://github.com/OpenGene/fastp)
* seqkit v2.3.1 #(https://bioinf.shenwei.me/seqkit/)
* PEAR v0.9.11 #(https://cme.h-its.org/exelixis/web/software/pear/doc.html)


# Directory structure

### source code

All code is in  <code>src/</code>, which contains a combination of bash, python, and R scripts. The numbering at the beginning of each file name indicates the order in which that script should be run.

### data
Due to the large size of output files related to sequencing data processing, these outputs are generally written to a separate directory: combicr_data/combicr_outputs/. This directory is outside of this repository. All other outputs are written to this repository in the  <code>data/</code> repository. The only exception are models (.pkl) or .csv files that are too large are not added to this repository. 



Raw sequencing data will be made publicly available under NCBI Bioproject PRJNA961693.

### plasmid sequences
Plasmid maps for key constructs used in this study can be found in the <code>plasmid_maps</code> directory. 

Please feel free to reach out if you have any questions about implementation or reproducing analyses! (alcantar [at] mit [dot] edu).
