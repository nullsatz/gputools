AC_INIT([gputools], 1.2) dnl package name, version
AC_CONFIG_AUX_DIR([tools])

dnl Select an optional include path, from a configure option
dnl or from an environment variable.
AC_ARG_WITH([cuda-include],
  AC_HELP_STRING([--with-cuda-include=INCLUDE_PATH],
    [the location of CUDA header files]),
  [cuda_include_path=$withval])
CUDA_CPPFLAGS="-I."
if test [ -n "${cuda_include_path}" ] ; then
  CUDA_CPPFLAGS="-I. -I${cuda_include_path}"
else
  if test [ -n "${CUDA_INCLUDE}" ] ; then
     CUDA_CPPFLAGS="-I. -I${CUDA_INCLUDE}"
     cuda_include_path="${CUDA_INCLUDE}"
  else
    if test [ -d "/usr/local/cuda/include" ] ; then
      CUDA_CPPFLAGS="-I. -I/usr/local/cuda/include"
      cuda_include_path="/usr/local/cuda/include"
    fi
  fi
fi

dnl ditto for a library path
AC_ARG_WITH([cuda-lib],
      AC_HELP_STRING([--with-cuda-lib=LIB_PATH],
         [the location of CUDA libraries]),
      [cuda_lib_path=$withval])
if test [ -n "${cuda_lib_path}" ] ; then
   LIBS="-L${cuda_lib_path} ${LIBS}"
else
  if test [ -n "${CUDA_LIBS}" ] ; then
     LIBS="-L${CUDA_LIBS} ${LIBS}"
     cuda_lib_path="${CUDA_LIBS}"
  else
    if test [ -d "/usr/local/cuda/lib64" ] ; then
      LIBS="-L/usr/local/cuda/lib64 ${LIBS}"
      cuda_lib_path="/usr/local/cuda/lib64"
    else
      if test [ -d "/usr/local/cuda/lib" ] ; then
        LIBS="-L/usr/local/cuda/lib ${LIBS}"
        cuda_lib_path="/usr/local/cuda/lib"
      fi
    fi
  fi
fi

dnl Now find the compiler and compiler flags to use
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CXX=`"${R_HOME}/bin/R" CMD config CXX`
CPP=`"${R_HOME}/bin/R" CMD config CPP`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
AC_PROG_CXX
AC_PROG_CPP

if test [ -n "${cuda_include_path}" ] ; then
  OLD_CPPFLAGS=${CPPFLAGS}
  CPPFLAGS="-I${cuda_include_path} ${CPPFLAGS}"
fi

AC_CHECK_HEADER([cuda.h], , [HAS_CUDA_H=no])
AC_CHECK_HEADER([nvrtc.h], , [HAS_NVRTC_H=no])

if test [ -n "${cuda_include_path}" ] ; then
  CPPFLAGS=${OLD_CPPFLAGS}
fi

if test [ "${HAS_CUDA_H}" = no ] ; then
  AC_MSG_ERROR("CUDA header cuda.h not found")
fi

if test [ "${HAS_NVRTC_H}" = no ] ; then
  AC_MSG_ERROR("CUDA header nvrtc.h not found")
fi

if test [ -n "${cuda_lib_path}" ] ; then
  OLD_LDFLAGS=${LDFLAGS}
  LDFLAGS="-L${cuda_lib_path} ${LDFLAGS}"
fi

dnl search for a library containing a CUDA function
AC_SEARCH_LIBS([cuGetErrorName], [cuda], ,
  AC_MSG_ERROR("no cuda driver library found"))

dnl search for a library containing a CUDA function
AC_SEARCH_LIBS([cudaMalloc], [cudart], ,
  AC_MSG_ERROR("no cuda runtime api library found"))

dnl search for a library containing a CUDA function
AC_SEARCH_LIBS([nvrtcCreateProgram], [nvrtc], ,
  AC_MSG_ERROR("no nvrtc library found"))

dnl search for a library containing a CUDA function
AC_SEARCH_LIBS([cublasSetMatrix], [cublas], ,
  AC_MSG_ERROR("no cublas library found"))

if test [ -n "${cuda_lib_path}" ] ; then
  LDFLAGS=${OLD_LDFLAGS}
  AC_CANONICAL_HOST
  case $host_os in
    darwin* )
      LIBS="${LIBS} -Wl,-rpath,${cuda_lib_path}"
      ;;
    * )
      LIBS="${LIBS} -Wl,-rpath=${cuda_lib_path}"
      ;;
  esac
fi

dnl substitute CUDA_CPPFLAGS and LIBS
AC_SUBST(CUDA_CPPFLAGS)
AC_SUBST(LIBS)
dnl and do substitution in the src/Makevars.in
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
