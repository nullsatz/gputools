# argument checks and small paths across the package
test_that("devices can be chosen and queried", {
  skip_if_no_gpu()
  expect_identical(chooseGpu(getGpuId()), getGpuId())
  expect_error(chooseGpu(-1), "not valid")
})

test_that("gpuMatMult and friends check dimensions", {
  a <- matrix(1:6, 2)
  expect_equal(cpuMatMult(a, t(a)), a %*% t(a))
  expect_error(cpuMatMult(a, a), "mismatched")
  skip_if_no_gpu()
  b <- matrix(rnorm(6), 2)
  expect_equal(gpuCrossprod(a, b), crossprod(a, b))
  expect_equal(gpuTcrossprod(a, b), tcrossprod(a, b))
  expect_error(gpuCrossprod(a, t(b)), "mismatch")
  expect_error(gpuTcrossprod(a, t(b)), "mismatch")
})

test_that("gpuDist and gpuHclust check their arguments", {
  pts <- matrix(rnorm(20), 10)
  expect_error(gpuDist(pts, "nope"), "distance method")
  expect_error(gpuHclust(dist(pts), "nope"), "clustering method")
  expect_error(gpuHclust(dist(pts[1, , drop = FALSE])), "n >= 2")
  expect_error(gpuDistClust(pts, distmethod = "nope"), "distance method")
  expect_error(gpuDistClust(pts, clustmethod = "nope"), "clustering method")
  skip_if_no_gpu()
  d <- gpuDist(pts, "euclidian")                 # the old spelling
  expect_equal(attr(d, "method"), "euclidean")
  expect_equal(attr(gpuDist(pts, "minkowski", p = 3), "p"), 3)
  expect_equal(gpuHclust(dist(pts))$labels, NULL)
})

test_that("other functions check their arguments", {
  x <- matrix(rnorm(20), 10)
  expect_error(gpuCor(x, method = "nope"), "correlation method")
  expect_error(gpuCor(x, use = "complete.obs"), "use must be")
  expect_error(gpuCor(x, x[1:5, ]), "different numbers of rows")
  expect_error(gpuQr(matrix(1i, 2, 2)), "complex")
  expect_error(gpuSolve(matrix(1i, 2, 2)), "complex")
  expect_error(gpuSvd(matrix(numeric(0), 0, 2)), "dimension is zero")
  expect_error(gpuGranger(x, lag = 4), "too short")
  expect_error(gpuPrcomp(cbind(x, 1), scale. = TRUE), "constant")
})

test_that("the declared routine objects are exactly the registered ones", {
  # R/globals.R declares them for CRAN's fake-install check; a routine added
  # to src/routines.h must be added there too
  registered <- names(getDLLRegisteredRoutines("gputools")$.Call)
  expect_setequal(gputools:::.gpuRoutines, paste0("C_", registered))
})
