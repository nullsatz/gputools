# the GPU column ranking behind Spearman's coefficient
gpuRank <- function(m, pr) .Call(gputools:::C_gpuRank, m, pr)
baseRank <- function(m) {
  r <- apply(m, 2L, rank, ties.method = "average", na.last = "keep")
  dim(r) <- dim(m)
  r
}

for (pr in precisions) {
  test_that(paste("GPU ranks match rank() in", pr), {
    skip_if_no_gpu()
    set.seed(15)
    for (n in c(1, 2, 7, 64, 65, 1000, 5000)) {
      m <- cbind(sample(1:5, n, replace = TRUE),        # many ties
                 rnorm(n),                              # none
                 round(rnorm(n), 1))                    # some
      storage.mode(m) <- "double"
      if (n > 3) m[sample(n, 3), 1] <- NA
      expect_equal(gpuRank(m, pr), baseRank(m), label = paste("n =", n))
    }
    allNA <- matrix(NA_real_, 5, 1)
    expect_equal(gpuRank(allNA, pr), baseRank(allNA))
  })
}
