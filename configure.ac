AC_INIT([gputools], 0.01)
AC_CONFIG_AUX_DIR([tools])

AC_MSG_NOTICE([checking for critical tools used by configure])

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
fi

DEVICEOPTS=""
OBJS13="kendall.o"

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
	;;
esac

LIBS="-L${CUDA_HOME}${CUDA_LIB_DIR} -lcublas"

AC_SUBST(LIBS)
AC_SUBST(CUDA_HOME)

AC_SUBST(CPICFLAGS)
AC_SUBST(DEVICEOPTS)
AC_SUBST(OBJS13)
AC_SUBST(RPATHFLAG)

AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT
