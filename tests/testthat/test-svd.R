set.seed(14)
tall <- matrix(rnorm(60 * 8), 60, 8)
wide <- t(tall[1:20, ])                  # 8 x 20

# singular vectors are unique only up to sign: align a's columns with b's
alignSigns <- function(a, b) sweep(a, 2, sign(colSums(a * b)), "*")

for (pr in precisions) {
  for (shape in c("tall", "wide")) {
    test_that(paste("gpuSvd matches svd() for a", shape, "matrix in", pr), {
      skip_if_no_gpu()
      x <- if (shape == "tall") tall else wide
      g <- gpuSvd(x, precision = pr)
      ref <- svd(x)
      expect_equal(g$d, ref$d, tolerance = tolFor(pr))
      expect_equal(alignSigns(g$u, ref$u), ref$u, tolerance = tolFor(pr) * 10)
      expect_equal(alignSigns(g$v, ref$v), ref$v, tolerance = tolFor(pr) * 10)
      expect_equal(g$u %*% diag(g$d) %*% t(g$v), x, tolerance = tolFor(pr))
    })
  }

  test_that(paste("gpuPrcomp matches prcomp() in", pr), {
    skip_if_no_gpu()
    for (sc in c(FALSE, TRUE)) {
      g <- gpuPrcomp(USArrests, scale. = sc, precision = pr)
      ref <- prcomp(USArrests, scale. = sc)
      expect_s3_class(g, "prcomp")
      expect_equal(g$sdev, ref$sdev, tolerance = tolFor(pr))
      expect_equal(alignSigns(g$rotation, ref$rotation), ref$rotation,
                   tolerance = tolFor(pr) * 10)
      expect_equal(alignSigns(g$x, ref$x), ref$x, tolerance = tolFor(pr) * 10)
      expect_equal(g$center, ref$center)
      expect_equal(g$scale, ref$scale)
      expect_equal(summary(g)$importance, summary(ref)$importance,
                   tolerance = 1e-3)
    }
  })
}

test_that("gpuSvd's nu and nv and gpuPrcomp's rank. and tol", {
  skip_if_no_gpu()
  g <- gpuSvd(tall, nu = 0, nv = 3, precision = "double")
  expect_null(g$u)
  expect_equal(dim(g$v), c(8L, 3L))
  expect_error(gpuSvd(tall, nu = 9), "cannot exceed")
  expect_equal(ncol(gpuPrcomp(tall, rank. = 2)$rotation), 2L)
  ref <- prcomp(tall, tol = 0.6)
  expect_equal(ncol(gpuPrcomp(tall, tol = 0.6, precision = "double")$rotation),
               ncol(ref$rotation))
  expect_error(gpuSvd(replace(tall, 3, NA)), "missing")
})
