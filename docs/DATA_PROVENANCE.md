# Data provenance and publication boundaries

## Water-quality observations

The harmonised site archive combines records obtained through Environment Agency, USGS, and NEON acquisition workflows. The local registry fixes the station identity, network, coordinates, parameter mapping, and time handling used by downstream stages. Source downloads and cleaned hourly shards are private archive material; their downloader and normalisation code is retained under `core/` and `scripts/`.

## Rainfall and calendar context

Rainfall was assembled from archived site precipitation plus audited API-derived inputs. Calendar features are deterministic transformations of timestamps. Every modelling transformation is fitted on training data only where a fitted statistic is required.

## External static site context

The completed 910-site audit contains four families:

| Family | Source/interpretation | Coverage |
|---|---|---:|
| Geology/alkalinity proxy | BGS 625K and USGS geologic maps; broad **site-point** lithology | 893 strict, 909 contextual |
| Baseflow index | GSCD BFI4 0.05-degree point-grid proxy | 910 |
| Imperviousness | GB Copernicus/EEA sealing; US NLCD; Alaska GHSL built-surface proxy | 904 direct, 910 including proxy |
| Permitted-discharge distance | Nearest active EA/NRW/SEPA/EPA location; straight-line/geodesic | 910 |

The all-four cohort is 893 sites under strict geology, 909 under contextual geology, and 899 under the conservative high-quality definition. The one unresolved geology site is `01579550`.

These fields do not establish causality. In particular, site-point geology is not a catchment-average alkalinity measurement, and distance to a permitted location is neither upstream-only nor proof of active pollutant loading.

## Upstream catchment context

The upstream audit uses GeoFRESH/hydrographr code with Hydrography90m v1.0, Environment90m, HydroLAKES, and their underlying climate, soil, elevation, and land-cover layers. The raw primary download is approximately 39.3 GB (589 verified files); the local `upstream1_primary` folder also contains indices and working material.

UPSTREAM13 contains area, elevation, gradient, precipitation, precipitation seasonality, six soil/bedrock fields, and urban/cropland/forest fractions. It does **not** contain upstream geology, carbonate fraction, alkalinity, aquifer permeability, or groundwater class. Aggregation is an equal-subcatchment mean of upstream subcatchment means, not an area-weighted land-cell catchment average.

The spatial product covers 886/910 usable sites, but seven fields were finite for 94.357% of usable sites. Because the frozen threshold was 95%, the audit stopped before candidate training.

## Solar context

The three solar fields are computed from timestamp, latitude, and longitude using `pvlib` solar position/NREL SPA, with `astral` used for independent checks. They are not observations downloaded from a solar-radiation service. The saved sidecars cover 910 sites and about 18.4 million hourly rows.

## Flow and stage

Dynamic river discharge was audited from official network sources, including native USGS `00060` discharge. USGS `00065` is stage (water level), not discharge. Coverage/support rules are encoded in the corresponding audit artefacts; neither the flow nor stage adapter was promoted as a final H00 improvement.

## Licensing and redistribution

- Keep raw EA and NRW permit extracts private; they can include personal or operator information and carry conditional terms.
- Publish only derived permit-distance fields that exclude identifying attributes, after checking the current licence.
- Preserve GSCD CC BY-NC and Hydrography90m/Environment90m non-commercial attribution requirements.
- Attribute BGS, Copernicus, USGS, GHSL, HydroLAKES, and original monitoring agencies in any derived publication.
- A file being technically copyable does not make it redistributable. The USB archive is a private preservation copy, not a public release bundle.
