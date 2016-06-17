AC_INIT([gputools], 1.1)
AC_CONFIG_AUX_DIR([tools])

AC_ARG_WITH([r],
  AC_HELP_STRING([--with-r=r],
    [specify the full path to the root of the R install directory, e.g. /usr/lib64/R]),
    [R_HOME="$withval"])

AC_ARG_WITH([r-include],
  AC_HELP_STRING([--with-r-include=r-include],
    [specify the full path to your R header files, e.g. /usr/share/R/include]),
    [R_INCLUDE="$withval"])

AC_ARG_WITH([r-lib],
  AC_HELP_STRING([--with-r-lib=r-lib],
    [specify the full path to your R shared libraries, e.g. /usr/lib64/R/lib]),
    [R_LIB="$withval"])

AC_ARG_WITH([cuda],
  AC_HELP_STRING([--with-cuda=cuda],
    [specify the root of your cuda install, e.g. /usr/local/cuda]),
    [CUDA_HOME="$withval"])

AC_ARG_WITH([nvcc],
  AC_HELP_STRING([--with-nvcc=nvcc],
    [specify a CUDA compiler, e.g. /usr/local/cuda/bin/nvcc]),
    [NVCC="$withval"])

AC_ARG_WITH([cuda-include],
  AC_HELP_STRING([--with-cuda-include=cuda-include],
    [specify the location of the CUDA header files, e.g. /usr/local/cuda/include]),
    [CUDA_INCLUDE="$withval"])

AC_ARG_WITH([cuda-lib],
  AC_HELP_STRING([--with-cuda-lib=cuda-lib],
    [specify the location of the CUDA shared libraries, e.g. /usr/local/cuda/lib64]),
    [CUDA_LIB="$withval"])

AC_MSG_CHECKING("the cuda compiler nvcc")
AC_PATH_PROG([NVCC], [nvcc],
             ["no"],
             [$PATH$PATH_SEPARATOR/usr/local/cuda/bin$PATH_SEPARATOR$CUDA_HOME/bin])
AS_IF([test "x$NVCC" = "xno"],
      [AC_MSG_ERROR(["NVCC compiler not found! Try --with-nvcc=/usr/local/cuda/bin/nvcc"])

AS_IF([test "x$R_INCLUDE" != "x"],
      [EXTRA_INCLUDE=" -I${R_INCLUDE} "])
AS_IF([test "x$R_LIB" != "x"],
      [EXTRA_LIB=" -L${R_LIB} "])

AS_IF([test "x$CUDA_INCLUDE" != "x"],
      [EXTRA_INCLUDE=" -I${CUDA_INCLUDE} "])
AS_IF([test "x$CUDA_LIB" != "x"],
      [EXTRA_LIB=" -L${CUDA_LIB} "])

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
  R_LIB="-L\"${R_LIB}\" -lR"
fi

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

CC=`"${R_HOME}/bin/R" CMD config CC`
CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`

if test "x$RPATH_FLAG" = "xgnu"; then
  RPATH="-Xlinker -rpath=\"${CUDA_LIB}\""
fi

if test "x$RPATH_FLAG" = "xllvm"; then
  RPATH="-Xlinker -rpath,\"${CUDA_LIB}\""
fi

if test "x$RPATH_FLAG" = "xnone"; then
  RPATH=""
fi

AC_CANONICAL_HOST
case $host_os in
  darwin* )
    LDFLAGS=`R CMD config --ldflags`
    LDFLAGS="-Xlinker \"${LDFLAGS}\""
    ;;
  *)
    LDFLAGS=""
    ;;
esac

AC_SUBST(R_HOME)
AC_SUBST(R_INCLUDE)
AC_SUBST(R_LIB)

AC_SUBST(NVCC)
AC_SUBST(CUDA_HOME)
AC_SUBST(CUDA_INCLUDE)
AC_SUBST(CUDA_LIB)

AC_SUBST(EXTRA_INCLUDES)
AC_SUBST(EXTRA_LIBS)

AC_SUBST(CC)
AC_SUBST(CFLAGS)
AC_SUBST(CPPFLAGS)
AC_SUBST(CPICFLAGS)

AC_SUBST(RPATH)
AC_SUBST(LDFLAGS)

AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT
