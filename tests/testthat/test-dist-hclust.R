set.seed(3)
pts <- matrix(rnorm(40 * 5), 40, 5)

for (m in c("euclidean", "maximum", "manhattan", "canberra", "minkowski")) {
  test_that(paste("gpuDist", m, "matches dist()"), {
    skip_if_no_gpu()
    expect_equal(as.vector(gpuDist(pts, m, p = 3)),
                 as.vector(dist(pts, m, p = 3)), tolerance = tol)
  })
}

test_that("gpuDist binary matches dist()", {
  skip_if_no_gpu()
  bin <- matrix(rbinom(40 * 8, 1, 0.4), 40, 8)
  expect_equal(as.vector(gpuDist(bin, "binary")),
               as.vector(dist(bin, "binary")), tolerance = tol)
})

for (m in c("single", "complete", "average", "mcquitty", "median",
            "centroid", "ward")) {
  test_that(paste("gpuHclust", m, "matches hclust() heights"), {
    skip_if_no_gpu()
    d <- dist(pts)
    # base R renamed "ward" to "ward.D"; the merge order can legitimately
    # differ on near-ties, the sequence of merge heights cannot
    base <- hclust(d, if (m == "ward") "ward.D" else m)
    expect_equal(sort(gpuHclust(d, m)$height), sort(base$height),
                 tolerance = tol)
  })
}

test_that("gpuDistClust is gpuDist then gpuHclust", {
  skip_if_no_gpu()
  expect_equal(gpuDistClust(pts)$height, gpuHclust(gpuDist(pts))$height,
               tolerance = tol)
})
