library("tictoc")

data_dir = "/home/julian/Uni/MasterThesis/psupplementary/data"
acinar_file = paste0(data_dir, "/acinar_sce.rda")
profile_out = "psupertime_rprofile_acinar.out"
topX = 20  # number of calls from profile to plot

# Plots bar graph with tilted x labels
rotate_x <- function(data, column_to_plot="total.time", main=NULL, rot_angle=35) {
  plt <- barplot(data[[column_to_plot]], main=main, col='steelblue', xaxt="n")
  text(plt, par("usr")[3], labels = rownames(data), srt = rot_angle, adj = c(1.1,1.1), xpd = TRUE, cex=0.6) 
}

# load SingleCellExperiment data  -> adds acinar_sce to namespace
load(acinar_file)

# Runs profiling 
tic(); Rprof(profile_out); psupertime::psupertime(acinar_sce, acinar_sce$donor_age); toc(); Rprof(NULL)

data <- summaryRprof(profile_out)$by.total
rotate_x(data[1:topX, ], main="Acinar SCE Rprofile (total time)")
