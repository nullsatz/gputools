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

AC_MSG_WARN([${CPPFLAGS}])

AC_CHECK_HEADER([cuda.h], [HAS_CUDA_H=yes], [HAS_CUDA_H=no])
AC_CHECK_HEADER([nvrtc.h], [HAS_NVRTC_H=yes], [HAS_NVRTC_H=no])

if test [ "${HAS_CUDA_H}" = "no" ] ; then
   if test [ -d "/usr/local/cuda/include" ] ; then
      CPPFLAGS="-I/usr/local/cuda/include ${OLD_CPPFLAGS}"
      AC_CHECK_HEADER([cuda.h], [HAS_CUDA_H=yes], [HAS_CUDA_H=no])
      if test [ "${HAS_CUDA_H}" = "yes" ] ; then
         cuda_include_path=/usr/local/cuda/include
         CUDA_CPPFLAGS="-I. -I${cuda_include_path}"
      fi
   fi
fi

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
AC_SEARCH_LIBS([cuGetErrorName], [cuda],
  [HAS_CUDA_LIB=yes],
  [HAS_CUDA_LIB=no])

dnl search for a library containing a CUDA function
AC_SEARCH_LIBS([cudaMalloc], [cudart],
  [HAS_CUDART_LIB=yes],
  [HAS_CUDART_LIB=no])

dnl search for a library containing a CUDA function
AC_SEARCH_LIBS([nvrtcCreateProgram], [nvrtc],
  [HAS_NVRTC_LIB=yes],
  [HAS_NVRTC_LIB=no])

dnl search for a library containing a CUDA function
AC_SEARCH_LIBS([cublasSetMatrix], [cublas],
  [HAS_CUBLAS_LIB=yes],
  [HAS_CUBLAS_LIB=no])

if test [ "${HAS_NVRTC_LIB}" = "no" ] ; then
  default_lib_dir=no
  if test [ -d "/usr/local/cuda/lib64" ] ; then
    default_lib_dir=/usr/local/cuda/lib64
  fi
  
  if test [ "${default_lib_dir}" = "no" ] ; then
    if test [ -d "/usr/local/cuda/lib" ] ; then
      default_lib_dir=/usr/local/cuda/lib
    fi
  fi
  
  if test [ "${default_lib_dir}" != "no" ] ; then
    LDFLAGS="-L${default_lib_dir} ${OLD_LDFLAGS}"

    AC_SEARCH_LIBS([cuGetErrorName], [cuda],
      [HAS_CUDA_LIB=yes],
      [HAS_CUDA_LIB=no])
    AC_SEARCH_LIBS([cudaMalloc], [cudart],
      [HAS_CUDART_LIB=yes],
      [HAS_CUDART_LIB=no])
    AC_SEARCH_LIBS([nvrtcCreateProgram], [nvrtc],
      [HAS_NVRTC_LIB=yes],
      [HAS_NVRTC_LIB=no])
    AC_SEARCH_LIBS([cublasSetMatrix], [cublas],
      [HAS_CUBLAS_LIB=yes],
      [HAS_CUBLAS_LIB=no])
  fi
  
  if test [ "${HAS_NVRTC_LIB}" = "yes" ] ; then
    cuda_lib_path="${default_lib_dir}"
    LIBS="-L${cuda_lib_path} ${LIBS}"
  fi
fi

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
