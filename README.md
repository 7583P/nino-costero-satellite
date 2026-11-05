# Niño Costero — Ingestor Satelital NOAA GOES-19 (10 Minutos, S/ 0)

Repositorio de adquisición y procesamiento incremental de observaciones meteorológicas satelitales en tiempo real para el proyecto **Niño Costero** (costa norte del Perú: Tumbes, Piura y Región Lambayeque).

## Misión Científica
Procesar escaneos auténticos del satélite geoestacionario NOAA GOES-19 con cadencia operacional de 10 minutos, bajo una arquitectura pública 24/7 sin costo alguno (presupuesto S/ 0.00).

## Fuentes de Datos Oficiales (Acceso Abierto NOAA NODD)
- **NOAA/NESDIS Enterprise Rain Rate (ERR)**: `s3://noaa-enterprise-rainrate-pds/G19/ABI-L2-FLDK-RainfallRate/` (tasa de precipitación instantánea en mm/h remuestreada a 2 km nativo).
- **GOES-19 ABI L2 CMI (Banda 13 - Infrarrojo de Onda Larga 10.3 µm)**: `s3://noaa-goes19/ABI-L2-CMIPF/`.
- **GOES-19 ABI L2 Cloud Products**:
  - `ABI-L2-ACMF` (Cloud Mask).
  - `ABI-L2-ACHA2KMF` (Cloud Top Height / Altura de cimas).
  - `ABI-L2-ACHTF` (Cloud Top Temperature / Temperatura de cimas).
  - `ABI-L2-ACTPF` (Cloud Top Phase / Fase termodinámica).
  - `ABI-L2-COD2KMF` (Cloud Optical Depth / Espesor óptico).
- **GOES-19 Geostationary Lightning Mapper (GLM L2 LCFA)**: Detección óptica de rayos en tiempo real sobre la cuenca del Pacífico y costa norte.

## Principios de Diseño
1. **Acceso Anónimo con Rango HTTP**: Utiliza solicitudes `HTTP Range` para leer únicamente los chunks espaciales correspondientes a la costa norte peruana (BBOX `[-10.2, -81.8, -3.0, -78.8]`), transfiriendo < 2 MB por archivo de 68 MB.
2. **Procesamiento Incremental**: Conserva el estado temporal entre ejecuciones consecutivas vía caché. Cada 10 minutos procesa únicamente la observación más reciente.
3. **Optimización de Cuota KV**: Deduplica imágenes por hash criptográfico SHA-256 e implementa una guarda de consumo diario (`DAILY_KV_WRITE_CAP = 750`) que garantiza operar estrictamente dentro del plan gratuito de Cloudflare KV (1,000 escrituras/día).
4. **Publicación Atómica y Segura**: Los activos PNG content-addressed se verifican individualmente antes de cometer el manifiesto `v3`. Se rechazan observaciones fuera de orden o desfasadas en el tiempo.
5. **Honestidad Científica**: Diferencia rigurosamente observaciones satelitales auténticas, reconstrucciones físicas volumétricas 3D y pronósticos numéricos.

## Estructura del Repositorio
```
├── .github/workflows/
│   └── satellite.yml      # Workflow recurrente cada 10 min (offset :07, :17, :27...)
├── data/
│   └── geo/               # Límites administrativos y localidades de referencia
├── sat/
│   ├── analysis.py        # Algoritmos de acumulación, índices de lluvia y tormenta
│   ├── encode.py          # Cuantización y codificación de activos PNG
│   ├── grid.py            # Proyección geoestacionaria ABI y coordenadas
│   ├── pipeline.py        # Ingestión, extracción regional y clasificación de nubes
│   ├── publish.py         # Publicador seguro con control de cuota KV
│   ├── rangeio.py         # Lector HTTP Range para AWS S3 NOAA
│   ├── sources.py         # Catálogos y parseo de NetCDF4/HDF5
│   └── validate.py        # Validación estricta del contrato de datos
├── tests/
│   └── test_sat_major.py  # Suite de verificación científica y regresión
├── requirements.txt       # Dependencias binarias fijadas
└── README.md
```

## Licencia
Código abierto bajo licencia MIT. Las observaciones satelitales provienen de NOAA Open Data Dissemination (dominio público). Límites geográficos: IGN Perú / OCHA / geoBoundaries (CC BY 3.0 IGO).

