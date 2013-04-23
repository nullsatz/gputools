AC_INIT([gputools], 0.01)
AC_CONFIG_AUX_DIR([tools])

AC_MSG_NOTICE([checking for critical tools used by configure])

AC_ARG_WITH([compiler-bindir],
	AC_HELP_STRING([--with-compiler-bindir=compiler_bindir],
		[specify the directory containing gcc, e.g. /usr/bin]),
	[COMPILER_BINDIR="$withval"])

AC_ARG_WITH([cuda-home],
	AC_HELP_STRING([--with-cuda-home=cuda_home],
		[specify the root of your cuda install, e.g. /usr/local/cuda]),
	[CUDA_HOME="$withval"])

AC_ARG_WITH([r-include],
	AC_HELP_STRING([--with-r-include=r_include],
		[specify the full path to your R header files, e.g. /usr/share/R/include]),
	[R_INCLUDE="$withval"])

AC_ARG_WITH([r-lib],
	AC_HELP_STRING([--with-r-lib=r-lib],
		[specify the full path to your R shared libraries, e.g. /usr/lib64/R/lib]),
	[R_LIB="$withval"])

AC_MSG_CHECKING("environment variable COMPILER_BINDIR")
if test -z "${COMPILER_BINDIR}"; then
	AC_MSG_RESULT("COMPILER_BINDIR not set; using user's default path")
else
	AC_MSG_RESULT("using COMPILER_BINDIR=${COMPILER_BINDIR}")
	AC_CHECK_FILE([${COMPILER_BINDIR}/gcc], [HAS_GCC="yes"])
	if test -z "${HAS_GCC}"; then
		AC_MSG_ERROR("cannot find gcc; check that ${COMPILER_BINDIR} is the correct location; using user's default path")
		MISC_PARAMS=""
	else
		AC_MSG_RESULT("gcc found in ${COMPILER_BINDIR}")
		MISC_PARAMS="--compiler-bindir ${COMPILER_BINDIR}"
	fi
fi

AC_MSG_CHECKING("environment variable CUDA_HOME")
if test -z "${CUDA_HOME}"; then
	AC_MSG_RESULT("CUDA_HOME not set; using default /usr/local/cuda")
	CUDA_HOME="/usr/local/cuda"
else
	AC_MSG_RESULT("using CUDA_HOME=${CUDA_HOME}")
fi

AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
if test -z "${HAS_NVCC}"; then
	AC_MSG_ERROR("cannot find nvcc; check CUDA install")
else
	AC_MSG_RESULT("nvcc found")
fi

AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [BIT64="yes"])
if test -z "${BIT64}"; then
	AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
	CUDA_LIB_DIR="/lib"
else
	AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
	CUDA_LIB_DIR="/lib64"
fi

${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
	AC_MSG_ERROR("could not determine R_HOME")
else
	CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`
	if test -z "${R_INCLUDE}"; then
		R_INCLUDE=`"${R_HOME}/bin/R" CMD config --cppflags`
	else
		R_INCLUDE="-I${R_INCLUDE}"
	fi
	if test -z "${R_LIB}"; then
		R_LIB=`"${R_HOME}/bin/R" CMD config --ldflags`
	else
		R_LIB="-L${R_LIB} -lR"
	fi
	AC_MSG_RESULT("using ${R_INCLUDE} for R header files")
	AC_MSG_RESULT("using ${R_LIB} for R shared libraries")
fi

DEVICEOPTS=""
OBJS13="kendall.o"
SVMDISABLED="FALSE"

AC_MSG_NOTICE([determining host platform])
arch=`uname -m`
osrelease=`uname -r`
osversion=`uname -v`

RPATHFLAG=""
AC_CANONICAL_HOST
case "${host_os}" in
	*darwin*)
		RPATHFLAG="-rpath,${CUDA_HOME}${CUDA_LIB_DIR}"
		RPATHFLAG="${RPATHFLAG},'${R_LIB}'"
		R_LIB=""
		AC_PATH_PROG([SW_VERS],[sw_vers],[no],[$PATH])
		if test "$arch" = "i386"; then
			$SW_VERS -productVersion | grep '^10.6' > /dev/null 2>&1
			if test "x$?" = "x0" ; then
				DEVICEOPTS="-m64"
			fi
		elif test "$arch" = "x86_64"; then
			DEVICEOPTS="-m64"
		else
			AC_MSG_WARN([Unrecognised Darwin Architecture])
		fi

		R_INCLUDE="${R_INCLUDE} ${RCPP_INCLUDE}"
		R_LIB="${R_LIB} ${RCPP_LIB}"
	;;
esac

AC_SUBST(LIBS)
AC_SUBST(MISC_PARAMS)
AC_SUBST(CUDA_HOME)

AC_SUBST(R_INCLUDE)
AC_SUBST(R_LIB)

AC_SUBST(CPICFLAGS)
AC_SUBST(DEVICEOPTS)
AC_SUBST(OBJS13)
AC_SUBST(SVMDISABLED)
AC_SUBST(RPATHFLAG)

AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT
