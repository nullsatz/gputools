// The whole package when configure found no usable CUDA (see
// configure.ac): every GPU entry point stops with an explanation of how to
// get a CUDA build, and the package still loads, so it can be installed on
// machines -- build servers, laptops -- that have no CUDA.

#include "R.h"
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

#include "nocuda-reason.h"
#include "routines.h"

static const char * message =
  "gputools was installed without CUDA (" GPUTOOLS_NO_CUDA_REASON "), "
  "so its GPU functions cannot run. To use them, install the NVIDIA CUDA "
  "toolkit (11.7 or later), make it findable -- set CUDA_HOME to its "
  "directory, or put nvcc on the PATH -- and reinstall gputools from "
  "source. See the package's INSTALL file.";

#define ARGS_0
#define ARGS_1 SEXP
#define ARGS_2 SEXP, SEXP
#define ARGS_3 SEXP, SEXP, SEXP
#define ARGS_4 SEXP, SEXP, SEXP, SEXP
#define ARGS_5 SEXP, SEXP, SEXP, SEXP, SEXP
#define ARGS_6 SEXP, SEXP, SEXP, SEXP, SEXP, SEXP
#define ARGS_7 SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP
#define ARGS_8 SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP
#define ARGS_9 SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP
#define ARGS_10 SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP

#define STUB(name, n) \
  SEXP name(ARGS_##n) { error("%s", message); return R_NilValue; }

extern "C" {

GPUTOOLS_GPU_ROUTINES(STUB)

SEXP gputoolsSetKernelDir(SEXP) { return R_NilValue; }
SEXP gputoolsUnloadKernels() { return R_NilValue; }

SEXP gputoolsBuildInfo()
{
  const char * names[] = { "cuda", "message", "" };
  SEXP out = PROTECT(mkNamed(VECSXP, names));
  SET_VECTOR_ELT(out, 0, ScalarLogical(FALSE));
  SET_VECTOR_ELT(out, 1, mkString(message));
  UNPROTECT(1);
  return out;
}

#define CALLDEF(name, n) {#name, (DL_FUNC) &name, n},

static const R_CallMethodDef callMethods[] = {
  GPUTOOLS_GPU_ROUTINES(CALLDEF)
  GPUTOOLS_OTHER_ROUTINES(CALLDEF)
  {NULL, NULL, 0}
};

void R_init_gputools(DllInfo * dll)
{
  R_registerRoutines(dll, NULL, callMethods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}

} // extern "C"
