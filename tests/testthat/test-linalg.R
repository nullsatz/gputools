set.seed(4)
x <- matrix(rnorm(30 * 6), 30, 6)
sq <- crossprod(x) + diag(6)
v <- rnorm(6)

for (pr in precisions) {
  test_that(paste("gpuQr reconstructs x and matches the rank in", pr), {
    skip_if_no_gpu()
    q <- gpuQr(x, precision = pr)
    expect_s3_class(q, "qr")
    expect_equal(q$rank, qr(x)$rank)
    expect_equal(qr.X(q), x, tolerance = tolFor(pr), ignore_attr = TRUE)
  })

  test_that(paste("gpuQr detects rank deficiency in", pr), {
    skip_if_no_gpu()
    xd <- cbind(x, x[, 1] + x[, 2])
    q <- gpuQr(xd, tol = 1e-4, precision = pr)
    expect_equal(q$rank, 6L)
    # column pivoting puts the diagonal of R in non-increasing magnitude
    r <- abs(diag(q$qr))[seq_len(q$rank)]
    expect_true(all(diff(r) <= 1e-5 * r[1]))
  })

  test_that(paste("gpuSolve matches solve() in", pr), {
    skip_if_no_gpu()
    expect_equal(gpuSolve(sq, precision = pr), solve(sq),
                 tolerance = tolFor(pr))
    expect_equal(gpuSolve(sq, v, precision = pr), solve(sq, v),
                 tolerance = tolFor(pr))
  })
}
