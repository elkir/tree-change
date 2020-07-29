
## Libraries
library(tidyverse)
# library(purrr)
# library(httr)
# library(magrittr)

library(raster)
library(rgdal)
library(lidR)

library(glue)
library(furrr)
# no_cores= availableCores()-1
future::plan(multiprocess,workers = 4)



years <- c(2013,2014)
# ws <-c(20)
ws <-c(15,16,17,18,19,21,22,23,24,25)


## Directories
dir_data <- "../Data/lidar/danum"

dir_chm <- paste(dir_data,"rasters",sep="/")
dir_treetops <- paste(dir_data,"treetops",years,sep="/")

dir_crowns <- paste(dir_data,"crowns","raster",years,sep = "/")
walk(dir_crowns,dir.create)

filename_crowns_index <- paste(dir_data,"index.txt",sep="/")
filename_crowns_errors <- paste(dir_data,"errors.txt",sep="/")



## Read rasters
chm_filenames <- paste0(dir_chm,"/raster",years,".tif")
chms <- map(chm_filenames,raster)

## Glue string formating
sprintf_transformer <- function(text, envir) {
  m <- regexpr(":.+$", text)
  if (m != -1) {
    format <- substring(regmatches(text, m), 2)
    regmatches(text, m) <- ""
    res <- eval(parse(text = text, keep.source = FALSE), envir)
    do.call(sprintf, list(glue("%{format}f"), res))
  } else {
    eval(parse(text = text, keep.source = FALSE), envir)
  }
}
glue_fmt <- function(..., .envir = parent.frame()) {
  glue(..., .transformer = sprintf_transformer, .envir = .envir)
}



### Functions

read_treetops <- function(year,ws){
  filename <-paste0("treetops_lmf_ws",ws,".shp") # SHP of JSON?
  filename_full <- paste(dir_data,"treetops",year,filename,sep = "/")
  readOGR(filename_full)
}


single_run <- function(iYear,ws,th_seed,th_cr,max_cr){
  year <- years[[iYear]]
  r <- chms[[iYear]]
  treetops <-read_treetops(year,ws)
  crowns <-dalponte2016(
    r,
    treetops,
    th_tree = 2,
    th_seed = th_seed,
    th_cr = th_cr,
    max_cr = max_cr,
    ID = "treeID"
  )()
  
  # write GeoTIFF to file and parameters to the index
  #FIXME ERROR!!!!
  # When rounding, it doesn't use the banker's rounding, rounding towards even.
  # This means it's inconsistent with Python loaders
  filename <- glue_fmt("dalponte_{year}_{ws}_seed{th_seed:.5}_cr{th_cr:.5}_max{max_cr:.3}.tif")
  #FIXME ^^^

  filename_full <- paste(dir_crowns[[iYear]],filename,sep = "/")
  print(filename_full)
  writeRaster(crowns,filename_full,overwrite=T)
}

double_run <- function(ws,th_seed,th_cr,max_cr){
  c(1,2) %>%
    future_map(~single_run(.,ws,th_seed,th_cr,max_cr))%>%
    {.}
  cat(c(ws,th_seed,th_cr,max_cr),"\n",file=filename_crowns_index,append=T)
  print(paste(c(ws,th_seed,th_cr,max_cr)))
}
random_run <- function() {
  th_seed = 1.0
  th_cr = 0.0
  # ensure seed threshold higher than cr threshold
  while (th_seed>th_cr){
    th_seed=runif(1,min=0.2,max=0.7)
    th_cr=runif(1,min=0.3,max=0.8)
  }
  max_cr=runif(1,min=40,max=70)
  ws_rand = sample(ws,1)
  try(
    double_run(ws=ws_rand,
               th_seed=th_seed,
               th_cr=th_cr,
               max_cr = max_cr),
    outFile = filename_crowns_errors
  )
}

## Parallel run
c(1:168) %>%
  future_map(~random_run())




