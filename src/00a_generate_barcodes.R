# example usage: Rscript 00a_generate_barcodes.R

library("DNABarcodes")

# create barcodes
mySetAshlock <- create.dnabarcodes(9, dist=5, heuristic="ashlock")

# create output directory
main_dir <- "../data/"
sub_dir <- "library_sequences/"
dir.create(file.path(main_dir, sub_dir), showWarnings = TRUE)
out_dir <- paste(main_dir,sub_dir, sep="")

# output file name
out_file_name <- "barcodes_len-9_dist-5.csv"

# save barcodes to csv file
write.csv(mySetAshlock,file= paste(out_dir, out_file_name, sep= ""), row.names=F)
