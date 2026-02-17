#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(soilDB)   # SDA_spatialQuery, get_SDA_property
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript scripts/ssurgo_fetch_clip_one_field.R <ffy_id> <bdry_gpkg> <out_gpkg> [vars_csv]",
       call. = FALSE)
}

ffy_id    <- args[[1]]
bdry_gpkg <- args[[2]]
out_gpkg  <- args[[3]]

vars_csv <- if (length(args) >= 4) args[[4]] else "sandtotal_r,silttotal_r,claytotal_r,awc_r,om_r,dbovendry_r"
vars <- unlist(strsplit(vars_csv, ",", fixed = TRUE))
vars <- trimws(vars)
vars <- vars[nzchar(vars)]

message("[SSURGO] ffy_id=", ffy_id)
message("[SSURGO] boundary=", bdry_gpkg)
message("[SSURGO] out=", out_gpkg)
message("[SSURGO] vars=", paste(vars, collapse=","))

# ---- read boundary ----
bdry <- st_read(bdry_gpkg, layer = "bdry", quiet = TRUE)
if (is.na(st_crs(bdry))) bdry <- st_set_crs(bdry, 4326)
bdry <- st_make_valid(bdry)
bdry_union <- st_union(bdry)

# ---- SDA spatial query: polygons intersecting boundary ----
res <- tryCatch(
  soilDB::SDA_spatialQuery(bdry_union, what="geom", db="SSURGO", geomIntersection=TRUE),
  error = function(e) NULL
)

# Fallback signature: aoi=
if (is.null(res)) {
  res <- tryCatch(
    soilDB::SDA_spatialQuery(aoi=bdry_union, what="geom", db="SSURGO", geomIntersection=TRUE),
    error = function(e) NULL
  )
}

if (is.null(res) || nrow(res) == 0) {
  stop("[SSURGO] SDA_spatialQuery returned 0 rows.", call. = FALSE)
}

# ---- find geometry column in result ----
geom_col <- NULL
for (cand in c("geom", "geometry", "wkt", "WKT", "wktgeom")) {
  if (cand %in% names(res)) { geom_col <- cand; break }
}
if (is.null(geom_col)) {
  stop(paste0("[SSURGO] Could not find geometry column in SDA result. Columns: ",
              paste(names(res), collapse=", ")), call. = FALSE)
}

# ---- build sf robustly ----
gobj <- res[[geom_col]]

if (inherits(gobj, "sfc")) {
  ss <- sf::st_as_sf(res)
  sf::st_geometry(ss) <- geom_col
} else {
  ss <- res %>%
    mutate(.geom = sf::st_as_sfc(.data[[geom_col]], crs = 4326)) %>%
    sf::st_as_sf(sf_column_name = ".geom")
}

# normalize column names to lower (helps with musym/muname variants)
names(ss) <- tolower(names(ss))

if (!("mukey" %in% names(ss))) stop("[SSURGO] SDA result missing 'mukey'.", call. = FALSE)

# keep useful descriptive cols if present
keep_desc <- intersect(c("areasymbol", "musym", "muname", "areaacres", "area_ac"), names(ss))

# ---- attach soil properties via SDA (mukey-level weighted avg) ----
mukeys <- unique(ss$mukey)

props <- tryCatch(
  soilDB::get_SDA_property(
    property = vars,
    method = "Weighted Average",
    mukeys = mukeys,
    top_depth = 0,
    bottom_depth = 150
  ),
  error = function(e) {
    message("[SSURGO] get_SDA_property failed: ", conditionMessage(e))
    NULL
  }
)

if (!is.null(props) && nrow(props) > 0) {
  names(props) <- tolower(names(props))
  # ensure mukey join key exists
  if (!("mukey" %in% names(props))) {
    stop("[SSURGO] get_SDA_property output missing 'mukey'.", call. = FALSE)
  }
  ss <- left_join(ss, props, by = "mukey")
} else {
  # create empty columns so schema is stable
  for (v in vars) {
    if (!(tolower(v) %in% names(ss))) ss[[tolower(v)]] <- NA_real_
  }
}

# ---- clip to boundary ----
ss <- st_set_crs(ss, 4326)
ss <- st_make_valid(ss)

ss_clip <- st_intersection(ss, bdry_union)
if (nrow(ss_clip) == 0) stop("[SSURGO] Intersection empty after clipping.", call. = FALSE)

# ---- area weights helper cols ----
# Use equal-area CRS for area computation
ss_5070 <- st_transform(ss_clip, 5070)
ss_clip$area_m2 <- as.numeric(st_area(ss_5070))
ss_clip$area_ac <- ss_clip$area_m2 / 4046.8564224

# ---- finalize columns ----
# Ensure geometry column name is "geometry"
sf::st_geometry(ss_clip) <- "geometry"

# Keep: mukey + descriptors + vars + area cols + geometry
vars_lower <- tolower(vars)
keep <- unique(c("mukey", keep_desc, vars_lower, "area_m2", "area_ac", "geometry"))
keep <- keep[keep %in% names(ss_clip)]

ss_out <- ss_clip %>% dplyr::select(dplyr::all_of(keep))

# ---- write gpkg ----
dir.create(dirname(out_gpkg), recursive = TRUE, showWarnings = FALSE)
if (file.exists(out_gpkg)) file.remove(out_gpkg)

st_write(ss_out, out_gpkg, layer = "ssurgo", quiet = TRUE)

message("[SSURGO] OK: wrote ", nrow(ss_out),
        " polygons, unique mukey=", length(unique(ss_out$mukey)),
        ", n_vars=", length(vars))
