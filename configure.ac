AC_INIT([gputools], [2.0.0])

dnl Locating CUDA, first match wins:
dnl   --with-cuda-include / --with-cuda-lib (each overrides its half)
dnl   --with-cuda-home
dnl   $CUDA_HOME, then $CUDA_PATH
dnl   the directory above nvcc on the PATH
dnl   /usr/local/cuda
dnl Headers and libraries are looked for in both the toolkit layout
dnl (include/, lib64/) and the conda layout (targets/<arch>/include, lib/).
dnl
dnl Without a usable CUDA the package is still built, from src/nocuda.cpp
dnl alone: it loads, says at attach time why it cannot use a GPU, and every
dnl GPU function stops with the same explanation. That is what a build
dnl server without CUDA (r-universe, CI) produces.

AC_ARG_WITH([cuda-home],
  AS_HELP_STRING([--with-cuda-home=DIR], [the CUDA toolkit installation]),
  [cuda_home=$withval])
AC_ARG_WITH([cuda-include],
  AS_HELP_STRING([--with-cuda-include=DIR], [the CUDA header files]),
  [cuda_include=$withval])
AC_ARG_WITH([cuda-lib],
  AS_HELP_STRING([--with-cuda-lib=DIR], [the CUDA libraries]),
  [cuda_lib=$withval])

if test -z "${cuda_home}"; then
  if test -n "${CUDA_HOME}"; then
    cuda_home=${CUDA_HOME}
  elif test -n "${CUDA_PATH}"; then
    cuda_home=${CUDA_PATH}
  else
    AC_PATH_PROG([NVCC], [nvcc])
    if test -n "${NVCC}"; then
      cuda_home=`dirname "${NVCC}"`
      cuda_home=`dirname "${cuda_home}"`
    elif test -d /usr/local/cuda; then
      cuda_home=/usr/local/cuda
    fi
  fi
fi

if test -z "${cuda_include}" && test -n "${cuda_home}"; then
  for d in "${cuda_home}/include" "${cuda_home}"/targets/*/include; do
    if test -f "${d}/cuda_runtime.h"; then
      cuda_include=${d}
      break
    fi
  done
fi

if test -z "${cuda_lib}" && test -n "${cuda_home}"; then
  for d in "${cuda_home}/lib64" "${cuda_home}/lib" "${cuda_home}"/targets/*/lib; do
    if ls "${d}"/libcudart.* >/dev/null 2>&1; then
      cuda_lib=${d}
      break
    fi
  done
fi

dnl no_cuda records why CUDA cannot be used; empty means it can.
no_cuda=

AC_MSG_CHECKING([for the CUDA headers])
if test -n "${cuda_include}" && test -f "${cuda_include}/cuda_runtime.h"; then
  AC_MSG_RESULT([${cuda_include}])
else
  AC_MSG_RESULT([no])
  no_cuda="the CUDA toolkit was not found"
fi
if test -z "${no_cuda}"; then
  AC_MSG_CHECKING([for the CUDA libraries])
  if test -n "${cuda_lib}"; then
    AC_MSG_RESULT([${cuda_lib}])
  else
    AC_MSG_RESULT([no])
    no_cuda="the CUDA runtime library (libcudart) was not found"
  fi
fi

if test -z "${no_cuda}"; then
  dnl Compile the checks with R's own compiler and flags.
  : ${R_HOME=`R RHOME`}
  if test -z "${R_HOME}"; then
    AC_MSG_ERROR([could not determine R_HOME])
  fi
  CXX=`"${R_HOME}/bin/R" CMD config CXX`
  CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`
  CPPFLAGS="-I${cuda_include} `"${R_HOME}/bin/R" CMD config CPPFLAGS`"
  LDFLAGS="-L${cuda_lib} `"${R_HOME}/bin/R" CMD config LDFLAGS`"
  AC_LANG([C++])

  AC_CHECK_HEADERS([cuda.h cuda_runtime.h nvrtc.h cublas_v2.h cusolverDn.h], ,
    [no_cuda="CUDA header $ac_header is missing from ${cuda_include}"
     break])
fi

if test -z "${no_cuda}"; then
  AC_MSG_CHECKING([for CUDA 11.7 or later])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <cuda_runtime_api.h>
#if CUDART_VERSION < 11070
#error too old
#endif]], [])],
    [AC_MSG_RESULT([yes])],
    [AC_MSG_RESULT([no])
     no_cuda="the CUDA toolkit in ${cuda_include} is older than 11.7"])
fi

if test -z "${no_cuda}"; then
  AC_SEARCH_LIBS([cudaGetDriverEntryPoint], [cudart], ,
    [no_cuda="no CUDA runtime library (libcudart) in ${cuda_lib}"])
fi
if test -z "${no_cuda}"; then
  AC_SEARCH_LIBS([nvrtcCreateProgram], [nvrtc], ,
    [no_cuda="no NVRTC library (libnvrtc) in ${cuda_lib}"])
fi
if test -z "${no_cuda}"; then
  AC_SEARCH_LIBS([cublasCreate_v2], [cublas], ,
    [no_cuda="no cuBLAS library (libcublas) in ${cuda_lib}"])
fi
if test -z "${no_cuda}"; then
  AC_SEARCH_LIBS([cusolverDnCreate], [cusolver], ,
    [no_cuda="no cuSOLVER library (libcusolver) in ${cuda_lib}"])
fi

if test -z "${no_cuda}"; then
  CUDA_CPPFLAGS="-I${cuda_include}"
  CUDA_LIBS="-L${cuda_lib} -Wl,-rpath,${cuda_lib} ${LIBS}"
  GPUTOOLS_OBJECTS=
  for f in src/*.cpp; do
    case "$f" in
      src/nocuda.cpp) ;;
      *) GPUTOOLS_OBJECTS="${GPUTOOLS_OBJECTS} `basename "$f" .cpp`.o" ;;
    esac
  done
else
  CUDA_CPPFLAGS=
  CUDA_LIBS=
  GPUTOOLS_OBJECTS=nocuda.o
  AC_MSG_WARN([

*** gputools is being installed WITHOUT CUDA: ${no_cuda}.
*** The package will load, but none of its GPU functions will run.
***
*** To use the GPU, install the NVIDIA CUDA toolkit (11.7 or later),
*** make it findable -- set CUDA_HOME to its directory, or put nvcc on
*** the PATH -- and reinstall gputools. See the INSTALL file.
])
fi
NO_CUDA_REASON=${no_cuda}

AC_SUBST([CUDA_CPPFLAGS])
AC_SUBST([CUDA_LIBS])
AC_SUBST([GPUTOOLS_OBJECTS])
AC_SUBST([NO_CUDA_REASON])
AC_CONFIG_FILES([src/Makevars src/nocuda-reason.h])
AC_OUTPUT
