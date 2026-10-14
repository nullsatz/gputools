.onLoad <- function(libname, pkgname) {
  # kernels are compiled from these sources on first use
  .Call(C_gputoolsSetKernelDir, file.path(libname, pkgname, "cuda"))
}

.onUnload <- function(libpath) {
  .Call(C_gputoolsUnloadKernels)
  library.dynam.unload("gputools", libpath)
}
