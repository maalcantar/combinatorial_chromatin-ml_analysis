# example usage: Rscript 00d_nupop_predictions.R

library("NuPoP")

# create output directory for results
main_dir_data <- "../data/"
sub_dir_results <- "nupop_results/"
dir.create(file.path(main_dir_data, sub_dir_results), showWarnings = TRUE)
out_dir_results <- paste(main_dir_data,sub_dir_results, sep="")

# create output directory for figs
main_dir_figs <- "../figs/"
sub_dir_figs <- "nupop_plots/"
dir.create(file.path(main_dir_figs, sub_dir_figs), showWarnings = TRUE)

# create function for renaming files (i.e., to move nupop results file)
# this is taken from: https://stackoverflow.com/questions/10266963/moving-files-between-folders
my.file.rename <- function(from, to) {
    todir <- dirname(to)
    if (!isTRUE(file.info(todir)$isdir)) dir.create(todir, recursive=TRUE)
    file.rename(from = from,  to = to)
}

# set up nupop parameters
species <- 7 # S. cerevisiae
model <- 4

reporter_sequence_dir <- "../data/reporter_sequences/"
# set paths to reporter sequences (in fasta format)
pMAA06_102_reporter_sequence <- paste0(reporter_sequence_dir,"pMAA06_102_reporter_sequence.fa")
pMAA06_109_reporter_sequence <- paste0(reporter_sequence_dir,"pMAA06_109_reporter_sequence.fa")
pMAA06_166_reporter_sequence <- paste0(reporter_sequence_dir,"pMAA06_166_reporter_sequence.fa")
pMAA06_167_reporter_sequence <- paste0(reporter_sequence_dir,"pMAA06_167_reporter_sequence.fa")
pMAA06_112_reporter_sequence <- paste0(reporter_sequence_dir,"pMAA06_112_reporter_sequence.fa")

# run nupop
predNuPoP(pMAA06_102_reporter_sequence, species, model)
predNuPoP(pMAA06_109_reporter_sequence, species, model)
predNuPoP(pMAA06_166_reporter_sequence, species, model)
predNuPoP(pMAA06_167_reporter_sequence, species, model)
predNuPoP(pMAA06_112_reporter_sequence, species, model)

nupop_results_suffix <- "_Prediction4.txt"
pMAA06_102_results_name <- paste0("pMAA06_102_reporter_sequence.fa", nupop_results_suffix)
pMAA06_109_results_name <- paste0("pMAA06_109_reporter_sequence.fa", nupop_results_suffix)
pMAA06_166_results_name <- paste0("pMAA06_166_reporter_sequence.fa", nupop_results_suffix)
pMAA06_167_results_name <- paste0("pMAA06_167_reporter_sequence.fa", nupop_results_suffix)
pMAA06_112_results_name <- paste0("pMAA06_112_reporter_sequence.fa", nupop_results_suffix)

pMAA06_102_results_name_final <- paste0(out_dir_results, pMAA06_102_results_name)
pMAA06_109_results_name_final <- paste0(out_dir_results, pMAA06_109_results_name)
pMAA06_166_results_name_final <- paste0(out_dir_results, pMAA06_166_results_name)
pMAA06_167_results_name_final <- paste0(out_dir_results, pMAA06_167_results_name)
pMAA06_112_results_name_final <- paste0(out_dir_results, pMAA06_112_results_name)

# move files to nupop output directory
my.file.rename(from=pMAA06_102_results_name,
               to=pMAA06_102_results_name_final)
my.file.rename(from=pMAA06_109_results_name,
               to=pMAA06_109_results_name_final)
my.file.rename(from=pMAA06_166_results_name,
               to=pMAA06_166_results_name_final)
my.file.rename(from=pMAA06_167_results_name,
               to=pMAA06_167_results_name_final)
my.file.rename(from=pMAA06_112_results_name,
               to=pMAA06_112_results_name_final)

# convert results to a dataframe and export as csv
pMAA06_102_results_df <- readNuPoP(pMAA06_102_results_name_final, startPos=1, endPos=5000)
pMAA06_109_results_df <- readNuPoP(pMAA06_109_results_name_final, startPos=1, endPos=5000)
pMAA06_166_results_df <- readNuPoP(pMAA06_166_results_name_final, startPos=1, endPos=5000)
pMAA06_167_results_df <- readNuPoP(pMAA06_167_results_name_final, startPos=1, endPos=5000)
pMAA06_112_results_df <- readNuPoP(pMAA06_112_results_name_final, startPos=1, endPos=5000)

write.csv(pMAA06_102_results_df, gsub('.txt', '.csv', pMAA06_102_results_name_final))
write.csv(pMAA06_109_results_df, gsub('.txt', '.csv', pMAA06_109_results_name_final))
write.csv(pMAA06_166_results_df, gsub('.txt', '.csv', pMAA06_166_results_name_final))
write.csv(pMAA06_167_results_df, gsub('.txt', '.csv', pMAA06_167_results_name_final))
write.csv(pMAA06_112_results_df, gsub('.txt', '.csv', pMAA06_112_results_name_final))
