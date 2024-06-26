library(BiocManager)
library(xcms)

#rsplit:
 # input:
 # x: String object that needs to be split up.
 # i: String used to split up the string of x.
 # output: 
 # Split up string

rsplit <- function(x, s ) {
  spl <- strsplit( x, s, fixed=TRUE )[[1]]
  res <- paste( spl[-length(spl)], collapse=s, sep="" )
  c( res, spl[length(spl)]  )
}

# If statement that checks if the user does not want to convert files in the pipeline.
if (snakemake@params[["xml_dir"]] != "dry_run") {
   # Generates a list of filenames gatherd from a ubuntu directory.
   files_names <- list.files(snakemake@params[["xml_dir"]])
   for (i in files_names){
     tryCatch({
     # Call the rsplit function to split the path so that only the exact file name remains.
     file_name <- rsplit(i,".")[1]
     # Create two strings that one contain the path where the .mzXML file needs to be pulled from,
     # and second a path where the .cdf needs to be saved.
     file_name <- paste(snakemake@params[["cdf_dir"]], "/", file_name, ".cdf",sep="")
     old_file_name <- paste(snakemake@params[["xml_dir"]], "/", rsplit(i,".")[1], ".mzXML",sep="")
     cat("filename:", file_name, "\n")
     # Convert raw GC-MS data into a xcms object, the xcms object can then be saved as a .cdf file using the
     # earlier made path.
     raw_data <- xcmsRaw(old_file_name)
     write.cdf(raw_data, file_name)}
     , error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
   }
}

