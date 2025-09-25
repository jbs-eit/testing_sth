# data_prep.R

library(here)
setwd(file.path(here(), 'sim-dashboard'))

dir.create("www/data", recursive = TRUE, showWarnings = FALSE)

library(jsonlite)

ages     <- c(40, 65)
scenarios<- c("baseline", "policy")

lookup <- list()
for (age in ages) {
  for (scn in scenarios) {
    set.seed(age + nchar(scn))
    # FAKE precomputed summary results
    df <- data.frame(
      year = 2000:2025,
      metric = cumsum(rnorm(26, mean = if (scn=="policy") 1 else 0.2, sd = 0.5)) + age/2,
      group = paste0("age", age, "_", scn)
    )
    path <- sprintf("www/data/results_age-%d_scn-%s.csv", age, scn)
    write.csv(df, path, row.names = FALSE)
    key <- paste(age, scn, sep="|")
    lookup[[key]] <- sub("^www/", "", path)  # store path relative to app root (e.g., "data/...")
  }
}

manifest <- list(
  age_values = ages,
  scenario_values = scenarios,
  lookup = lookup
)
writeLines(toJSON(manifest, auto_unbox = TRUE, pretty = TRUE), "index.json")
cat("Wrote CSVs and index.json\n")