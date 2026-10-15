# gpuKmeans against kmeans(algorithm = "Lloyd"): from the same starts, the
# same clusters, centres, sums of squares and iteration counts. The data
# are well-separated blobs, so no point is near-equidistant from two
# centres and rounding cannot send it the other way.

blobs <- function(seed, n = 600, d = 3, k = 5, offset = 0) {
  set.seed(seed)
  mids <- matrix(rnorm(k * d, sd = 10), k, d)
  x <- mids[rep(seq_len(k), length.out = n), ] +
    matrix(rnorm(n * d, sd = 0.5), n, d)
  x + offset
}

sameAsKmeans <- function(g, r, pr) {
  expect_equal(g$cluster, r$cluster)
  expect_equal(g$size, r$size)
  expect_equal(g$iter, r$iter)
  expect_equal(g$centers, r$centers, tolerance = tolFor(pr))
  expect_equal(g$withinss, r$withinss, tolerance = tolFor(pr))
  expect_equal(g$tot.withinss, r$tot.withinss, tolerance = tolFor(pr))
  expect_equal(g$totss, r$totss)
  expect_equal(g$betweenss, r$betweenss, tolerance = tolFor(pr))
}

for (pr in precisions) {
  test_that(paste("gpuKmeans matches kmeans Lloyd from given starts in", pr), {
    skip_if_no_gpu()
    # far from the origin too, where |c|^2 - 2 x.c would cancel uncentred
    for (offset in c(0, 1e4)) {
      x <- blobs(61, offset = offset)
      starts <- x[c(1, 7, 13, 300, 450), ]   # some clusters start twice
      r <- kmeans(x, starts, algorithm = "Lloyd", iter.max = 50)
      g <- gpuKmeans(x, starts, iter.max = 50, precision = pr)
      expect_s3_class(g, "kmeans")
      sameAsKmeans(g, r, pr)
    }
  })

  test_that(paste("gpuKmeans picks the same random starts as kmeans in", pr), {
    skip_if_no_gpu()
    x <- blobs(62)
    for (nstart in c(1L, 4L)) {
      set.seed(9)
      r <- kmeans(x, 5, nstart = nstart, algorithm = "Lloyd", iter.max = 50)
      set.seed(9)
      g <- gpuKmeans(x, 5, nstart = nstart, iter.max = 50, precision = pr)
      sameAsKmeans(g, r, pr)
    }
  })
}

test_that("gpuKmeans works through many clusters in blocks", {
  skip_if_no_gpu()
  # k = 2000 makes a block 33,554 points, so 40,000 points take two. One
  # pass: the first assignment and the means it gives. Uniform data has
  # near-ties, so a few points may go the other way.
  set.seed(63)
  x <- matrix(runif(40000 * 2), ncol = 2)
  starts <- x[seq_len(2000), ]
  r <- suppressWarnings(kmeans(x, starts, algorithm = "Lloyd", iter.max = 1))
  g <- suppressWarnings(gpuKmeans(x, starts, iter.max = 1,
                                  precision = "double"))
  expect_gt(mean(g$cluster == r$cluster), 0.999)
  expect_equal(g$iter, r$iter)
  expect_equal(sum(g$size), nrow(x))
})

test_that("gpuKmeans warns and refuses as kmeans does", {
  skip_if_no_gpu()
  x <- blobs(64)
  expect_warning(gpuKmeans(x, x[c(1, 2, 3, 4, 5), ], iter.max = 1),
                 "did not converge in 1 iteration")
  far <- rbind(x[1:4, ], c(1e6, 1e6, 1e6))
  expect_warning(g <- gpuKmeans(x, far, precision = "double"),
                 "empty cluster")
  expect_equal(g$size[5], 0L)
  expect_true(all(is.nan(g$centers[5, ])))

  rownames(x) <- paste0("p", seq_len(nrow(x)))
  colnames(x) <- c("a", "b", "c")
  g <- gpuKmeans(x, x[1:5, ])
  expect_equal(names(g$cluster), rownames(x))
  expect_equal(dimnames(g$centers), list(as.character(1:5), c("a", "b", "c")))
  expect_equal(fitted(g, "classes"), g$cluster)

  expect_error(gpuKmeans(x, x[c(1, 1), ]), "not distinct")
  expect_error(gpuKmeans(x[1:2, ], x[1:3, ]), "more cluster centers")
  expect_error(gpuKmeans(x, x[1:5, 1:2]), "same number of columns")
  expect_error(gpuKmeans(x, 3, iter.max = 0), "positive")
  x[1, 1] <- NA
  expect_error(gpuKmeans(x, 3), "missing")
})
