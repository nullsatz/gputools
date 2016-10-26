AC_INIT([gputools], 1.1)
AC_CONFIG_AUX_DIR([tools])

AC_ARG_WITH([r],
  AC_HELP_STRING([--with-r=r],
    [path to r install directory, e.g. /usr/lib64/R]),
    [R_HOME="$withval"])

AC_ARG_WITH([r-include],
  AC_HELP_STRING([--with-r-include=r-include],
    [path to r header files, e.g. /usr/share/R/include]),
    [R_INCLUDE="$withval"])

AC_ARG_WITH([r-lib],
  AC_HELP_STRING([--with-r-lib=r-lib],
    [path to r libs, e.g. /usr/lib64/R/lib]),
    [R_LIB="$withval"])

AC_ARG_WITH([cuda],
  AC_HELP_STRING([--with-cuda=cuda],
    [path to cuda toolkit, e.g. /usr/local/cuda]),
    [CUDA_HOME="$withval"])

AC_ARG_WITH([nvcc],
  AC_HELP_STRING([--with-nvcc=nvcc],
    [path of cuda compiler, e.g. /usr/local/cuda/bin/nvcc]),
    [NVCC="$withval"])

AC_ARG_WITH([cuda-include],
  AC_HELP_STRING([--with-cuda-include=cuda-include],
    [path to cuda header files, e.g. /usr/local/cuda/include]),
    [CUDA_INCLUDE="$withval"])

AC_ARG_WITH([cuda-lib],
  AC_HELP_STRING([--with-cuda-lib=cuda-lib],
    [path to cuda libs, e.g. /usr/local/cuda/lib64]),
    [CUDA_LIB="$withval"])

AC_MSG_CHECKING("the cuda compiler nvcc")
AC_PATH_PROG([NVCC], [nvcc],
             ["no"],
             [$PATH$PATH_SEPARATOR/usr/local/cuda/bin$PATH_SEPARATOR$CUDA_HOME/bin])
AS_IF([test "x$NVCC" = "xno"],
      [AC_MSG_ERROR(["nvcc compiler not found! Try --with-nvcc=/usr/local/cuda/bin/nvcc"])])


AC_MSG_CHECKING("R")

if test -z "${R_HOME}"; then
  : ${R_HOME=`R RHOME`}
  if test -z "${R_HOME}"; then
    echo "could not determine R_HOME"
    exit 1
  fi
fi
AC_MSG_RESULT("using ${R_HOME} for the root of the R install directory")

if test -z "${R_INCLUDE}"; then
  R_INCLUDE="${R_HOME}/include"
fi
AC_MSG_RESULT("using ${R_INCLUDE} for R header files")

if test -n "${R_LIB}"; then
  AC_MSG_RESULT("using ${R_LIB} for R shared libraries")
fi

AS_IF([test "x$R_INCLUDE" != "x"],
      [EXTRA_INCLUDES="${EXTRA_INCLUDES} -I${R_INCLUDE} "])
AS_IF([test "x$R_LIB" != "x"],
      [EXTRA_LIBS="${EXTRA_LIBS} -L${R_LIB} -lR "])

AS_IF([test "x$CUDA_INCLUDE" != "x"],
      [EXTRA_INCLUDES="${EXTRA_INCLUDES} -I${CUDA_INCLUDE} "])
AS_IF([test "x$CUDA_LIB" != "x"],
      [EXTRA_LIBS="${EXTRA_LIBS} -L${CUDA_LIB} -lcublas -lnvrtc -lcuda "])

AC_MSG_CHECKING([for rpath flag style])
AC_PROG_CC([cc cl gcc])
saved_cflags="${CFLAGS}"
CFLAGS="-Wl,--rpath=."
AC_LINK_IFELSE([AC_LANG_SOURCE([int main(){ return 0;}])],
  [RPATH_FLAG=gnu],
  [RPATH_FLAG=none])

if test "x$RPATH_FLAG" = "xnone"; then
  CFLAGS="-Wl,-rpath,."
  AC_LINK_IFELSE([AC_LANG_SOURCE([int main(){ return 0;}])],
    [RPATH_FLAG=llvm],
    [RPATH_FLAG=none])
fi
CFLAGS="${saved_cflags}"
AC_MSG_RESULT([rpath flag style... ${RPATH_FLAG}])

CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`

EXTRA_CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`
EXTRA_CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`

RPATH=""
if test "x$CUDA_LIB" != "x"; then
if test "x$RPATH_FLAG" = "xgnu"; then
  RPATH="-rpath=\"${CUDA_LIB}\""
fi

if test "x$RPATH_FLAG" = "xllvm"; then
  RPATH="-rpath,\"${CUDA_LIB}\""
fi

if test "x$RPATH_FLAG" = "xnone"; then
  RPATH=""
fi
fi

AC_CANONICAL_HOST
case $host_os in
  darwin* )
    DARWIN_LDFLAGS=`R CMD config --ldflags`
    ;;
  *)
    DARWIN_LDFLAGS=""
    ;;
esac

EXTRA_LDFLAGS="${RPATH} ${DARWIN_LDFLAGS} ${EXTRA_LDFLAGS}"

if test "x$CUDA_LIB" = "x"; then
  EXTRA_LIBS="${EXTRA_LIBS} -lcublas -lnvrtc -lcuda "
fi

AC_SUBST(NVCC)

AC_SUBST(EXTRA_CFLAGS)
AC_SUBST(EXTRA_CPPFLAGS)
AC_SUBST(EXTRA_LDFLAGS)

AC_SUBST(EXTRA_INCLUDES)
AC_SUBST(EXTRA_LIBS)

AC_SUBST(CPICFLAGS)

AC_SUBST(RPATH)

AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT
