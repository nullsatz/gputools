# .C() marks single-precision results with a "Csingle" attribute. Left on a
# returned object it makes base R's own .C/.Fortran calls (qr.qy(), hclust(),
# ...) allocate float buffers and then write doubles into them.
hasCsingle <- function(obj) {
  if (!is.null(attr(obj, "Csingle"))) return(TRUE)
  if (is.list(obj)) return(any(vapply(obj, hasCsingle, logical(1))))
  FALSE
}

set.seed(7)
x <- matrix(rnorm(40 * 5), 40, 5)
sq <- crossprod(x) + diag(5)

test_that("no returned object carries the Csingle attribute", {
  skip_if_no_gpu()
  d <- data.frame(y = rnorm(40), a = x[, 1], b = x[, 2])
  results <- list(
    gpuCor = gpuCor(x),
    gpuCorKendall = gpuCor(x, method = "kendall"),
    gpuTtest = gpuTtest(c(20, 30), c(0.1, 0.5)),
    gpuDist = gpuDist(x),
    gpuHclust = gpuHclust(dist(x)),
    gpuDistClust = gpuDistClust(x),
    gpuQr = gpuQr(x),
    gpuSolve = gpuSolve(sq),
    gpuSolveVec = gpuSolve(sq, rnorm(5)),
    gpuGranger = gpuGranger(x, lag = 1),
    gpuMi = gpuMi(x),
    gpuLm = gpuLm(y ~ a + b, d),
    gpuLsfit = gpuLsfit(x, d$y),
    gpuGlm = gpuGlm(y ~ a + b, data = d)
  )
  for (nm in names(results))
    expect_false(hasCsingle(results[[nm]]), label = nm)
})
