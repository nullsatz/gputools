set.seed(1)
a <- matrix(rnorm(60 * 40), 60, 40)
b <- matrix(rnorm(40 * 30), 40, 30)

test_that("gpuMatMult matches %*%", {
  skip_if_no_gpu()
  expect_equal(gpuMatMult(a, b), a %*% b, tolerance = 1e-10)
})

test_that("gpuCrossprod and gpuTcrossprod match base", {
  skip_if_no_gpu()
  expect_equal(gpuCrossprod(a), crossprod(a), tolerance = 1e-10)
  expect_equal(gpuTcrossprod(b), tcrossprod(b), tolerance = 1e-10)
})

test_that("mismatched dimensions are an error", {
  expect_error(gpuMatMult(a, a), "mismatched")
})
