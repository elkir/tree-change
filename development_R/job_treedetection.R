library(tidyverse)
# library(purrr)
# library(httr)
# library(magrittr)

library(raster)
library(rgdal)
library(lidR)

library(functional)
library(RColorBrewer)

library(furrr)
future::plan(multiprocess)

# directories
dir_data <-"/home/lune/Data/ai4er/mres/lidar"
dir_2013 <- paste(dir_data, "ETH_2013_data",sep="/")
dir_2014 <- paste(dir_data, "NERC_2014_data",sep="/")
dir_2013_raster <-  paste(dir_2013,"CHM_raster",sep="/")
dir_2014_raster <-  paste(dir_2014,"CHM_raster",sep="/")
filenames_2013 <- list.files(dir_2013_raster, full.names=TRUE)
filenames_2014 <- list.files(dir_2014_raster, full.names=TRUE)


## Build rasters
# raster2013 <- do.call(merge,lapply(filenames_2013, raster))
# raster2014 <- do.call(merge,lapply(filenames_2014, raster))
# extIntersect  <- intersect(raster2013@extent,raster2014@extent)
# rasters <- future_map(rasters,Curry(crop,y=extIntersect))

## Load rasters
raster2013 = raster(paste(dir_data,"raster","raster2013.tif",sep="/"))
raster2014 = raster(paste(dir_data,"raster","raster2014.tif",sep="/"))


rasters <- c("2013"=raster2013,"2014"=raster2014)

# define functions
write_treetops_parameters <- function(data, year,ws) {
  dirname <- paste(dir_data,"vector","treetops",year,sep="/")
  filename <- paste0(dirname,"/","treetops_lmf_ws",ws,".json")
  writeOGR(data,
           filename,
           paste0(year,"_ws",ws),
           driver="GeoJSON")
}


single_run <- function(year,ws){
  treetops <-tree_detection(rasters[[year]],lmf(ws))
  write_treetops_parameters(treetops,year,ws)
}

### Parameters
## Run1
# years <- c("2013","2014")
# sizes <- c(15:25)
## Run2
# years <- c("2014")
# sizes <- c(15:16)
## Run3
# years <- c("2013")
# sizes <- c(16)
## Run4
years <- c("2014")
sizes <- c(15,16,19,24)


# prepare output directories
dir_output_treetops <- paste(dir_data,"vector","treetops",years,sep="/")
walk(dir_output_treetops,~dir.create(., recursive=T))

## paralelize years
double_run <- function(ws) {
  years %>%
    future_map(~single_run(.,ws))
}

## run all the WS
sizes %>%
  future_map(double_run,.progress=T)


