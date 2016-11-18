.onLoad <- function(lib, pkg) {
  cuFileNames <-
    c( "correlation.cu"
     , "distance.cu"
     , "granger.cu"
     , "hcluster.cu"
     , "kendall.cu"
     , "mi.cu"
     , "qrdecomp.cu"
     )

  cuFileNames <-
    sapply(cuFileNames,
      function(fn) {
        system.file('cuda', fn, package = 'gputools')
      })

  cuSrc <-
    sapply(cuFileNames,
      function(fn) {
        readChar(fn, file.info(fn)$size)
      })

  cuFiles <-
    c( "correlation"
     , "distance"
     , "granger"
     , "hcluster"
     , "kendall"
     , "mi"
     , "qrdecomp"
     )

  result <-
    .C("cuCompile",
      length(cuFiles),
      cuFiles,
      cuSrc)
}

.unLoad <- function(lib, pkg) {
  result <- .C("unloadPackage")
}
