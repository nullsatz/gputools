.onLoad <- function(libname, pkgname) {
  # kernels are compiled from these sources on first use
  .Call(C_gputoolsSetKernelDir, file.path(libname, pkgname, "cuda"))
}

.onAttach <- function(libname, pkgname) {
  info <- .Call(C_gputoolsBuildInfo)
  if (!info$cuda) packageStartupMessage(info$message)
}

.onUnload <- function(libpath) {
  .Call(C_gputoolsUnloadKernels)
  library.dynam.unload("gputools", libpath)
}
