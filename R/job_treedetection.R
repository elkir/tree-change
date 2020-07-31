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

#TODO genearalize for any data folder and any years


# directories
dir_data <-"../Data/lidar/paracou"
#dir_2013 <- paste(dir_data,"original_data", "ETH_2013_data",sep="/")
#dir_2014 <- paste(dir_data,"original_data", "NERC_2014_data",sep="/")
#dir_2013_raster <-  paste(dir_2013,"CHM_raster",sep="/")
#dir_2014_raster <-  paste(dir_2014,"CHM_raster",sep="/")
##filenames_2013 <- list.files(dir_2013_raster, full.names=TRUE)
##filenames_2014 <- list.files(dir_2014_raster, full.names=TRUE)
##
##
#### Build rasters
## raster2013 <- do.call(merge,lapply(filenames_2013, raster))
## raster2014 <- do.call(merge,lapply(filenames_2014, raster))
## extIntersect  <- intersect(raster2013@extent,raster2014@extent)
## rasters <- future_map(rasters,Curry(crop,y=extIntersect))

## Load rasters
raster2016 <- raster(paste(dir_data, "rasters", "raster2016.tif", sep="/"))
raster2019 <- raster(paste(dir_data, "rasters", "raster2019.tif", sep="/"))


rasters <- c("2016"=raster2016,"2019"=raster2019)

# define functions
write_treetops_parameters <- function(data, year,ws) {
  dirname <- paste(dir_data,"treetops",year,sep="/")
  filename <- paste0(dirname,"/","treetops_lmf_ws",ws)
  writeOGR(data,
           filename,
           paste0(year,"_ws",ws),
           , driver="ESRI Shapefile")
}


single_run <- function(year,ws){
  treetops <-tree_detection(rasters[[year]],lmf(ws))
  write_treetops_parameters(treetops,year,ws)
}

### Parameters
# Run1
 years <- c("2016","2019")
 sizes <- c(17:24)

## Run2
# years <- c("2019")
# sizes <- c(15:16)
## Run3
# years <- c("2016")
# sizes <- c(16)
## Run4
#years <- c("2019")
#sizes <- c(15,16,19,24)


# prepare output directories
dir_output_treetops <- paste(dir_data,"treetops",years,sep="/")
walk(dir_output_treetops,~dir.create(., recursive=T))

## paralelize years
double_run <- function(ws) {
  print(ws)
  years %>%
    map(~single_run(.,ws))
}

## run all the WS
sizes %>%
  map(double_run)


