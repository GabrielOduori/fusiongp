library(raster)
library(leaflet)
library(viridis)
library(viridisLite)

BASE_NO2 <- raster("/home/gabriel-oduori/Dropbox/EPA/EPA/NO2_Mean_ATMO-Street_2023.tif")

# Reproject to WGS84 lon/lat for leaflet
BASE_NO2_ll <- projectRaster(BASE_NO2, crs = CRS("EPSG:4326"), method = "bilinear")

# Palette for raster values
pal <- colorNumeric(
  palette = rev(rainbow(256)),
  domain  = values(BASE_NO2_ll),
  na.color = "transparent"
)

# Helper: shrink extent inward by a fraction (e.g., 0.03 = 3% padding inwards)
pad_extent <- function(e, frac = -0.03) {
  dx <- (e@xmax - e@xmin) * frac
  dy <- (e@ymax - e@ymin) * frac
  extent(
    e@xmin + dx,
    e@xmax - dx,
    e@ymin + dy,
    e@ymax - dy
  )
}

pal_rev <- colorNumeric(
  palette = rev(viridis(256)),
  domain  = values(BASE_NO2_ll),
  na.color = "transparent"
)

e <- extent(BASE_NO2_ll)
e_pad <- pad_extent(e, frac = 0.01)  # change to 0.01, 0.05, etc. to taste

leaflet(
  width  = "100%",
  height = "100vh"
) %>%
  addProviderTiles("Stadia.StamenToner") %>%
  addRasterImage(BASE_NO2_ll, colors = pal, opacity = 0.9) %>%
  # fitBounds(e_pad@xmin, e_pad@ymin, e_pad@xmax, e_pad@ymax) %>%
  setView(
    lng  = mean(c(e_pad@xmin, e_pad@xmax)),
    lat  = mean(c(e_pad@ymin, e_pad@ymax)),
    zoom = 12
  ) %>%
  addLegend(
    position = "topright",
    pal = pal_rev,
    values = values(BASE_NO2_ll),
    title   = "NO\u2082 (\u03BCg/m\u00B3)",
    opacity = 1
  )


# CALL THE MAP
m