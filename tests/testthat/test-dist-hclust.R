set.seed(3)
pts <- matrix(rnorm(40 * 5), 40, 5)

for (pr in precisions) {
  for (m in c("euclidean", "maximum", "manhattan", "canberra", "minkowski")) {
    test_that(paste("gpuDist", m, "matches dist() in", pr), {
      skip_if_no_gpu()
      expect_equal(as.vector(gpuDist(pts, m, p = 3, precision = pr)),
                   as.vector(dist(pts, m, p = 3)), tolerance = tolFor(pr))
    })
  }

  test_that(paste("gpuDist binary matches dist() in", pr), {
    skip_if_no_gpu()
    bin <- matrix(rbinom(40 * 8, 1, 0.4), 40, 8)
    expect_equal(as.vector(gpuDist(bin, "binary", precision = pr)),
                 as.vector(dist(bin, "binary")), tolerance = tolFor(pr))
  })

  for (m in c("single", "complete", "average", "mcquitty", "median",
              "centroid", "ward")) {
    test_that(paste("gpuHclust", m, "matches hclust() heights in", pr), {
      skip_if_no_gpu()
      d <- dist(pts)
      # base R renamed "ward" to "ward.D"; the merge order can legitimately
      # differ on near-ties, the sequence of merge heights cannot
      base <- hclust(d, if (m == "ward") "ward.D" else m)
      expect_equal(sort(gpuHclust(d, m, precision = pr)$height),
                   sort(base$height), tolerance = tolFor(pr))
    })
  }

  for (m in c("single", "complete", "average", "mcquitty", "wpgma", "median",
              "centroid", "ward", "flexible", "flexible group")) {
    test_that(paste("gpuHclust", m, "follows the Lance-Williams update in",
                    pr), {
      skip_if_no_gpu()
      d <- dist(pts)
      expect_equal(sort(gpuHclust(d, m, precision = pr)$height),
                   sort(lanceWilliams(d, m)), tolerance = tolFor(pr))
    })
  }

  test_that(paste("gpuDistClust is gpuDist then gpuHclust in", pr), {
    skip_if_no_gpu()
    expect_equal(gpuDistClust(pts, precision = pr)$height,
                 gpuHclust(gpuDist(pts, precision = pr),
                           precision = pr)$height,
                 tolerance = tolFor(pr))
  })
}

test_that("non-finite input is an error, not garbage", {
  skip_if_no_gpu()
  bad <- pts
  bad[2, 3] <- NA
  expect_error(gpuDist(bad), "NA/NaN/Inf")
})

test_that("precision must be single or double", {
  expect_error(gpuDist(pts, precision = "half"), "single.*double")
})

for (pr in precisions) {
  for (m in c("flexible", "flexible group")) {
    test_that(paste("gpuHclust", m, "takes beta in", pr), {
      skip_if_no_gpu()
      d <- dist(pts)
      for (b in c(-0.25, 0, 0.5)) {
        expect_equal(sort(gpuHclust(d, m, beta = b, precision = pr)$height),
                     sort(lanceWilliams(d, m, lambda = b, beta = b)),
                     tolerance = tolFor(pr), label = paste("beta", b))
        expect_equal(gpuDistClust(pts, clustmethod = m, beta = b,
                                  precision = pr)$height,
                     gpuHclust(gpuDist(pts, precision = pr), m, beta = b,
                               precision = pr)$height,
                     tolerance = tolFor(pr))
      }
    })
  }
}

test_that("beta must be below 1", {
  expect_error(gpuHclust(dist(pts), "flexible", beta = 1), "below 1")
  expect_error(gpuHclust(dist(pts), "flexible", beta = NA), "below 1")
})
