set.seed(12)
sq <- crossprod(matrix(rnorm(40 * 8), 40, 8)) + diag(8)
dimnames(sq) <- list(paste0("r", 1:8), paste0("c", 1:8))
v <- rnorm(8)
B <- matrix(rnorm(8 * 3), 8, 3, dimnames = list(NULL, c("p", "q", "r")))
tall <- matrix(rnorm(30 * 5), 30, 5)
yt <- rnorm(30)

for (pr in precisions) {
  test_that(paste("gpuSolve matches solve() in", pr), {
    skip_if_no_gpu()
    expect_equal(gpuSolve(sq, precision = pr), solve(sq),
                 tolerance = tolFor(pr))
    expect_equal(gpuSolve(sq, v, precision = pr), solve(sq, v),
                 tolerance = tolFor(pr))
    expect_equal(gpuSolve(sq, B, precision = pr), solve(sq, B),
                 tolerance = tolFor(pr))
  })

  test_that(paste("gpuSolve solves an overdetermined x by least squares in",
                  pr), {
    skip_if_no_gpu()
    expect_equal(gpuSolve(tall, yt, precision = pr),
                 qr.coef(qr(tall), yt), tolerance = tolFor(pr))
    # its "inverse" is the pseudo-inverse
    expect_equal(gpuSolve(tall, precision = pr) %*% tall, diag(5),
                 tolerance = tolFor(pr), ignore_attr = TRUE)
  })

  test_that(paste("gpuSolve refuses singular systems in", pr), {
    skip_if_no_gpu()
    sing <- sq
    sing[, 3] <- 0
    expect_error(gpuSolve(sing, v, precision = pr), "singular")
    dep <- cbind(tall, tall[, 1] + tall[, 2])
    expect_error(gpuSolve(dep, yt, precision = pr), "rank deficient")
  })
}

test_that("gpuSolve checks its arguments", {
  expect_error(gpuSolve(t(tall)), "underdetermined")
  expect_error(gpuSolve(sq, rnorm(5)), "nrows")
})
