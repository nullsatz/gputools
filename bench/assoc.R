# gpuAssoc against Matrix eQTL (Shabalin 2012), the CPU program it follows:
# an eQTL scan of every SNP against every gene, linear model, five
# covariates, reporting p <= 1e-5. Matrix eQTL comes from CRAN; without it,
# only the GPU is timed.
#
#   Rscript bench/assoc.R [output directory]

library(gputools)
stopifnot(gpuAvailable())
out <- commandArgs(trailingOnly = TRUE)
out <- if (length(out)) out[1] else "vignettes"
haveME <- requireNamespace("MatrixEQTL", quietly = TRUE)

once <- function(f) { gc(); system.time(f())[["elapsed"]] }
samples <- 500
rows <- list()
for (sz in list(c(2000, 5000), c(10000, 5000), c(10000, 20000),
                c(50000, 20000), c(200000, 20000))) {
  nsnp <- sz[1]; ngene <- sz[2]
  set.seed(1)
  snps <- matrix(as.double(rbinom(samples * nsnp, 2, 0.3)), samples, nsnp)
  cv <- matrix(rnorm(samples * 5), samples, 5)
  genes <- matrix(rnorm(samples * ngene), samples, ngene)
  for (g in seq_len(ngene / 100))                      # planted eQTLs
    genes[, g] <- genes[, g] + 0.4 * snps[, sample(nsnp, 1)]

  for (pr in c("single", "double"))                       # warm up
    invisible(gpuAssoc(snps[, 1:100], genes[, 1:100], cv, precision = pr))
  found <- NA
  gpu <- once(function() found <<- nrow(gpuAssoc(snps, genes, cv,
                                                  pvalue = 1e-5)))
  gpuDouble <- if (nsnp <= 10000) once(function()
    gpuAssoc(snps, genes, cv, pvalue = 1e-5, precision = "double")) else NA

  me <- NA; meFound <- NA
  # beyond a billion tests Matrix eQTL needs more memory than the
  # benchmark job has (its sliced copies of the data)
  if (haveME && nsnp * as.double(ngene) <= 1e9) {
    sd <- function(m) { s <- MatrixEQTL::SlicedData$new()
                        s$CreateFromMatrix(t(m)); s }
    S <- sd(snps); G <- sd(genes); C <- sd(cv)
    me <- once(function() {
      res <- MatrixEQTL::Matrix_eQTL_engine(S, G, C,
        output_file_name = tempfile(), pvOutputThreshold = 1e-5,
        useModel = MatrixEQTL::modelLINEAR, verbose = FALSE,
        pvalue.hist = FALSE, min.pv.by.genesnp = FALSE,
        noFDRsaveMemory = FALSE)
      meFound <<- res$all$neqtls
    })
  }
  cat(sprintf("snps %7d genes %6d  tests %.1e  found %6d (MatrixEQTL %s)  MatrixEQTL %8.2f  gpu %7.2f  gpu double %7.2f\n",
              nsnp, ngene, nsnp * as.double(ngene), found,
              format(meFound), me, gpu, gpuDouble))
  rows[[length(rows) + 1]] <- data.frame(snps = nsnp, genes = ngene,
    samples = samples, covariates = ncol(cv), found = found,
    foundMatrixEQTL = meFound, MatrixEQTL = me, single = gpu,
    double = gpuDouble)
}
write.csv(do.call(rbind, rows), file.path(out, "assoc.csv"), row.names = FALSE)
